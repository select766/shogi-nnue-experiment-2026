@echo off
setlocal

rem Windows launcher for YaneuraOu-expert-blending.exe.
rem It mirrors the bash launcher by passing engine stdout through tee:
rem   engine stdout -> GUI stdout
rem                 -> log file for visualization tools

set "SCRIPT_DIR=%~dp0"
set "ENGINE=%SCRIPT_DIR%YaneuraOu-expert-blending.exe"

if not exist "%ENGINE%" (
    echo Error: engine binary not found: %ENGINE% 1>&2
    exit /b 1
)

rem onnxruntime.dll is placed in bin next to the engine. Put this directory
rem first so GUI-launched processes can find it regardless of their CWD.
set "PATH=%SCRIPT_DIR%;%PATH%"

rem Override from the GUI or wrapper if a visualizer watches another path.
if not defined YANEURAOU_EXPERT_BLENDING_LOG (
    set "YANEURAOU_EXPERT_BLENDING_LOG=%TEMP%\yaneuraou-expert-blending.log"
)

"%ENGINE%" %* | powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$path=$env:YANEURAOU_EXPERT_BLENDING_LOG; $writer=$null; $fs=$null; try { $dir=[IO.Path]::GetDirectoryName($path); if ($dir -and -not [IO.Directory]::Exists($dir)) { [IO.Directory]::CreateDirectory($dir) | Out-Null }; $enc=New-Object System.Text.UTF8Encoding($false); $fs=[IO.File]::Open($path,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::ReadWrite); $writer=New-Object IO.StreamWriter($fs,$enc) } catch { $writer=$null; if ($fs) { $fs.Dispose(); $fs=$null } }; try { foreach ($line in $input) { [Console]::Out.WriteLine($line); if ($writer) { try { $writer.WriteLine($line); $writer.Flush() } catch { $writer.Dispose(); $writer=$null } } } } finally { if ($writer) { $writer.Dispose() } elseif ($fs) { $fs.Dispose() } }"
