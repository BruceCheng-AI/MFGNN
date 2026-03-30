@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%run_mfgnn.py" ^
  --meta-epochs 1 ^
  --fine-tune-epochs 1 ^
  --num-tasks 1 ^
  --task-batch-size 1 ^
  --batch-size 2 ^
  --device cpu ^
  --save-interval 1 ^
  %*

endlocal
