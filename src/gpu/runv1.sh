#!/bin/bash

set -eo pipefail
mkdir -p ../../bin
mkdir -p ../../results/gpu

outputFrames=false
version=1
blkDim=8
extraSpan=12
showHelp=false
while getopts ":v:b:s:gh" opt; do
	case ${opt} in 
	  v) version=$OPTARG;;	
	  b) blkDim=$OPTARG;;
	  s) extraSpan=$OPTARG;;
	  g) outputFrames=true;;
	  h) showHelp=true;;
	  \?) echo "Usage: cmd -v <version> [-b <blkDim>] [-s <extraSpan>] [-g]";;
	esac
done
shift "$((OPTIND - 1))"

if $showHelp;then
    printf "\nFlags:
    -v : version of the code
    -b : blk dimension to run with
    -s : extra span to run with
    -g : generate the output frames 
    -h : help\n\n"
    exit
fi

resultsDir="../../results/gpu"
ouputFile="${resultsDir}/v${version}.txt"
previousVersion=0
outputProgram=../../bin/gpu_v$version
if $outputFrames ;then
	nvcc -arch sm_52 ../common/utils.c ../common/block.c \
		../common/prediction_frame.c mainv1.cu -DOUTPUT_FRAMES -o $outputProgram
else
	nvcc -arch sm_52 ../common/utils.c ../common/block.c \
		../common/prediction_frame.c mainv1.cu -o $outputProgram
fi

resultsTxtPath=../../results/gpu/v${version}.txt
printf "[\n   Version= $version\n   BlkDim= $blkDim\n   ExtraSpan= $extraSpan\n   OuputFrames= $outputFrames\n]\n"
printf "[ Version= $version, BlkDim= $blkDim, ExtraSpan=$extraSpan, OuputFrames= $outputFrames ]\n\n" > $resultsTxtPath

echo "Running on Foreman..."
$outputProgram ../../frames/ForemanYF4.yuv ../../frames/ForemanYF1.yuv \
	../../results/gpu/foreman_v${version}.yuv $blkDim $extraSpan 352 288 >> $resultsTxtPath

echo "Running on Jockey..." 
$outputProgram ../../frames/JockeyYF2.yuv ../../frames/JockeyYF1.yuv \
	../../results/gpu/jockey_v${version}.yuv $blkDim $extraSpan 3840 2160 >> $resultsTxtPath

echo "Running on Beauty..."
$outputProgram ../../frames/BeautyYF2.yuv ../../frames/BeautyYF1.yuv \
	../../results/gpu/beauty_v${version}.yuv $blkDim $extraSpan 3840 2160 >> $resultsTxtPath

if $outputFrames ;then
	printf "Output YUV file dimensions:\n   Foreman:\t(352 x 1440)\n   Jockey:\t(3840 x 10,800)\n   Beauty:\t(3840 x 10,800)\n"
fi

if [[ $version -gt 1 ]];then
	previousVersion=$((version-1))
fi

if [[ $previousVersion  -eq 0 ]];then
	exit 0
fi

count=0
images="Foreman Jockey Beauty"
imageTokens=( $images )
metrics="totalTime CPU->GPU kernel GPU->CPU PSNR"
metricTokens=( $metrics )
epsilon=0.1

while read -r current && read -r previous <&3;do
	currentVals=( $current )gi
	previousVals=( $previous )
	printf "Regression Testing: ${imageTokens[$count]}\n"
	for i in {0..4};do
		currentVal=${currentVals[$i]}
		previousVal=${previousVals[$i]}
		if (( $(echo "$currentVal > ($previousVal + $epsilon)" | bc -l) ));then
			printf "  Worse for ${metricTokens[$i]}\tcurrent: $currentVal\tpreviousVal: $previousVal\n"
		fi
	done
	count=$((count + 1))
done < <(tail -n3 "${resultsDir}/v${version}.txt") 3< <(tail -n3 "${resultsDir}/v$previousVersion.txt")

