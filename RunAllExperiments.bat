@ECHO OFF

@REM ----------------Digits Experiments----------------
set SamplesPerClass=1,3,5,7
set Methods="SDA_IO" "CCSA" "dSNE"

@REM USPS->MNIST
(for %%s in (%SamplesPerClass%) do (
    (for %%X IN (0,1,10) do (
        (for %%m in (%Methods%) do (
            python RunExperiment.py --Src "U" --Tgt "M" --SamplesPerClass  %%s  --Method %%m --GPU_ID 0 --LogToWandb True
        ))
     ))
))

@REM MNIST->USPS
(for %%s in (%SamplesPerClass%) do (
    (for %%X IN (0,1,10) do (
        (for %%m in (%Methods%) do (
            python RunExperiment.py --Src "M" --Tgt "U" --SamplesPerClass  %%s  --Method %%m --GPU_ID 0 --LogToWandb True
        ))
     ))
))


@REM ----------------Office Experiments----------------
(for %% IN (0,1,5) do (
    (for %%m in (%Methods%) do (
        python RunExperiment.py --Src "A" --Tgt "W" --SamplesPerClass  3  --Method %%m --GPU_ID 0 --LogToWandb True
        python RunExperiment.py --Src "A" --Tgt "D" --SamplesPerClass  3  --Method %%m --GPU_ID 0 --LogToWandb True
        python RunExperiment.py --Src "W" --Tgt "A" --SamplesPerClass  3  --Method %%m--GPU_ID 0 --LogToWandb True
        python RunExperiment.py --Src "W" --Tgt "D" --SamplesPerClass  3  --Method %%m --GPU_ID 0 --LogToWandb True
        python RunExperiment.py --Src "D" --Tgt "A" --SamplesPerClass  3  --Method %%m --GPU_ID 0 --LogToWandb True
        python RunExperiment.py --Src "D" --Tgt "W" --SamplesPerClass  3  --Method %%m --GPU_ID 0 --LogToWandb True
    ))
))
