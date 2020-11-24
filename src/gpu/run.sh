#!/bin/bash
mkdir -p ../../bin
mkdir -p ../../results/gpu

outputFrames=false
version=1
blkDim=16
extraSpan=7
while getopts ":v:b:s:g" opt; do
	case ${opt} in 
	  v) version=$OPTARG;;	
	  b) blkDim=$OPTARG;;
	  s) extraSpan=$OPTARG;;
	  g) outputFrames=true;;
	  \?) echo "Usage: cmd -v <version> [-b <blkDim>] [-s <extraSpan>] [-g]";;
	esac
done
echo $version $outputFrames $blkDim $extraSpan
shift "$((OPTIND - 1))"

resultsDir="../../results/gpu"
ouputFile="${resultDir}/v${outputID}.txt"
previousVersion=0

if $outputFrames ;then
	echo "Will output frames"
	nvcc -arch sm_52 YUVreadfile.cu YUVwritefile.cu ../common/block.c ../common/prediction_frame.c GPUBaseline.cu -DOUTPUT_FRAMES -o ../../bin/gpu
else
	echo "Won't output frames"
	nvcc -arch sm_52 YUVreadfile.cu YUVwritefile.cu ../common/block.c ../common/prediction_frame.c GPUBaseline.cu -o ../../bin/gpu
fi

echo "BlkDim: ${blkDim}, ExtraSpan: ${extraSpan}"
echo "Running on Foreman..."
../../bin/gpu ../../frames/ForemanYF4.yuv ../../frames/ForemanYF1.yuv \
	../../results/gpu/foreman_${blkDim}_${extraSpan} $blkDim $extraSpan 352 288 > $outputFile

echo "Running on Jockey..." 
../../bin/gpu ../../frames/JockeyYF2.yuv ../../frames/JockeyYF1.yuv \
	../../results/gpu/jockey_${blkDim}_${extraSpan} $blkDim $extraSpan 3840 2160 >> $outputFile

echo "Running on Beauty..."
../../bin/gpu ../../frames/BeautyYF2.yuv ../../frames/BeautyYF1.yuv \
	../../results/gpu/beauty_${blkDim}_${extraSpan} $blkDim $extraSpan 3840 2160 >> $outputFile

if [[ $version -gt 1 ]];then
	previousVersion=$((version-1))
fi

if [[ $previousVersion  -eq 0 ]];then
	exit 0
fi

count=0
images="Foreman Jockey Beauty"
imageTokens=( $images )
metrics="totalTime CPU->GPU GPU->CPU kernel"
metricTokens=( $metrics )
epsilon=0.001

while read -r current && read -r previous <&3;do
	currentVals=( $current )
	previousVals=( $previous )
	printf "Image: ${imageTokens[$count]}\n"
	for i in {0..3};do
		currentVal=${currentVals[$i]}
		previousVal=${previousVals[$i]}
		if (( $(echo "$currentVal > ($previousVal + $epsilon)" | bc -l) ));then
			printf "  Worse for ${metricTokens[$i]}\tcurrent: $currentVal\tpreviousVal: $previousVal\n"
		fi
	done
	count=$((count + 1))
done < $outputFile 3<"${resultDir}/$previousVersion.txt"


