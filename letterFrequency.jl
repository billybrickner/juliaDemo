#! /usr/bin/env julia

using Base64: base64encode, base64decode
using BenchmarkTools
using Test
using CUDA

function getFrequencies()::Tuple{UInt32, UInt32, Vector{Float64}}
    letterScores  = [
    ['A',50.0778966131907],
    ['B',30.2092691622103],
    ['C',40.8846702317291],
    ['D',23.1073083778966],
    ['E',24.6778966131907],
    ['F',17.9591800356506],
    ['G',16.6153297682709],
    ['H',22.0377896613191],
    ['I',39.8060606060606],
    ['J',14.0295900178253],
    ['K',8.3030303030303],
    ['L',19.0702317290553],
    ['M',46.2520499108734],
    ['N',36.6147950089127],
    ['O',18.8413547237077],
    ['P',25.7110516934046],
    ['Q',2.07825311942959],
    ['R',26.1048128342246],
    ['S',54.3620320855615],
    ['T',58.0146167557932],
    ['U',10.2474153297683],
    ['V',5.53529411764706],
    ['W',19.1078431372549],
    ['X',1.35080213903743],
    ['Y',16.8087344028521],
    ['Z',1.0],
    ['a',938.285026737968],
    ['b',154.395008912656],
    ['c',349.449554367201],
    ['d',422.427807486631],
    ['e',1380.00748663102],
    ['f',231.180926916221],
    ['g',215.106417112299],
    ['h',526.890909090909],
    ['I',807.011051693405],
    ['j',11.7390374331551],
    ['k',82.1368983957219],
    ['l',455.107308377897],
    ['m',261.564349376114],
    ['n',808.47504456328],
    ['o',843.006417112299],
    ['p',223.810873440285],
    ['q',9.6650623885918],
    ['r',737.602317290553],
    ['s',746.204991087344],
    ['t',981.763279857397],
    ['u',287.579857397504],
    ['v',116.465240641711],
    ['w',181.04385026738],
    ['x',22.0279857397504],
    ['y',189.311942959002],
    ['z',11.8401069518717],
    [' ',2279.00118835413]]
    minChar = typemax(UInt32)
    maxChar = typemin(UInt32)

    for letterScore in letterScores
        letter, score = letterScore
        charVal = UInt32(letter)
        if charVal < minChar
            minChar = charVal
        end
        if charVal > maxChar
            maxChar = charVal
        end
    end

    scores = Float64[0.0 for i in minChar:maxChar]
    for letterScore in letterScores
        letter, score = letterScore
        scores[Int(letter)-minChar+1] = score
    end

    return minChar, maxChar, scores
end

function cpuMain(cipherText)
    minChar, frequencies = getFrequencies()
end

function getKeyOracle():Char
    return '🔥'
end

global const MAX_VALUE_CHECKED = 0x80000000

function getKey(secretText::Vector{UInt32})::Char
    lowestChar, highestChar, frequencies = getFrequencies()
    maxScore = UInt32(0)
    maxKey = Char(0)
    for key in typemin(UInt32):MAX_VALUE_CHECKED
        score = 0
        for c in secretText[1:64]
            cGuess = c ⊻ key
            if (lowestChar <= cGuess) && (highestChar >= cGuess)
                score += frequencies[(cGuess - lowestChar) + 1]
            end
        end
        if score > maxScore
            maxScore = score
            maxKey = key
        end
    end
    return Char(maxKey)
end

function getKeyThreaded(secretText::Vector{UInt32})::Char
    lowestChar, highestChar, frequencies = getFrequencies()
    maxScore = UInt32(0)
    maxKey = Char(0)
    lk = ReentrantLock()
    Threads.@threads for thrdIdx in 1:Threads.nthreads()
        threadMaxScore = maxScore
        threadMaxKey = maxKey
        for key in typemin(UInt32):Threads.nthreads():MAX_VALUE_CHECKED
            key = key + thrdIdx - 1
            score = UInt32(0)
            for c in secretText
                cGuess = c ⊻ key
                if (lowestChar <= cGuess) && (highestChar >= cGuess)
                    @inbounds score += frequencies[(cGuess - lowestChar) + 1]
                end
            end
            if score > threadMaxScore
                threadMaxScore = score
                threadMaxKey = key
            end
        end
        if threadMaxScore > maxScore
            lock(lk) do
                if threadMaxScore > maxScore
                    maxScore = threadMaxScore
                    maxKey = threadMaxKey
                end
            end
        end
    end
    return Char(maxKey)
end

function getKeyGPUKernel!(secretText::CuDeviceVector{UInt32, 1},
                            keyScores::CuDeviceVector{Float32, 1},
                            frequencies::CuDeviceVector{Float32,1},
                            lowestChar::UInt32,
                            highestChar::UInt32)
    index = threadIdx().x
    stride = blockDim().x
    for keyIndex = index:stride:length(keyScores)
        score = UInt32(0)
        for c in secretText
            key = keyIndex - 1
            cGuess = c ⊻ key
            if (lowestChar <= cGuess) && (highestChar >= cGuess)
                @inbounds score += frequencies[(cGuess - lowestChar) + 1]
            end
        end
        @inbounds keyScores[keyIndex] = score
    end
    return nothing
end

function getKeyGPU(secretText::Vector{UInt32})::Char
    lowestChar, highestChar, frequencies = getFrequencies()
    keyScores = CUDA.fill(0.0f0, MAX_VALUE_CHECKED+1)
    @cuda  threads=1024 getKeyGPUKernel!(
                            cu(secretText),
                            keyScores,
                            cu(frequencies),
                            lowestChar,
                            highestChar)
    maxScore, maxCharIndex = findmax(keyScores)
    return Char(maxCharIndex - 1)
end

function main()
    open("secret.txt") do f
        secretCodedText = ""
        for line in eachline(f)
            secretCodedText *= line
        end
        secretText = base64decode(secretCodedText)
        secretText = Vector(reinterpret(UInt32, secretText))
        secretTextSub = secretText[1:64]
        key = getKeyOracle()
        print("Normal Loop          ")
        @test key == @btime getKey($secretTextSub)
        print("Threaded (", Threads.nthreads(), " threads)")
        @test key == @btime getKeyThreaded($secretTextSub)
        print("GPU                  ")
        @test key == @btime getKeyGPU($secretTextSub)
        text = [Char(UInt32(c) ⊻ UInt32(key)) for c in secretText]
        println("Bounds:")
        println("\tMin: 0x00000000")
        println("\tMax: 0x80000000")
        println("Key: ", key)
        println("Text:")
        println(join(text[1:57]))
    end
end

main()

