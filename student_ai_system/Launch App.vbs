Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the folder where this script lives
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Show startup message
MsgBox "MIT College Student AI System is starting..." & vbCrLf & vbCrLf & _
       "1. Installing dependencies (first run only)" & vbCrLf & _
       "2. Loading ML models" & vbCrLf & _
       "3. Opening browser at http://127.0.0.1:5000" & vbCrLf & vbCrLf & _
       "Click OK and wait 5-10 seconds for the browser to open.", _
       64, "MIT College — Student AI System"

' Run the batch file
WshShell.CurrentDirectory = scriptDir
WshShell.Run "cmd /k START_WINDOWS.bat", 1, False
