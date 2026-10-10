Option Explicit

Dim fileSystem, shell, scriptDirectory, powerShellExecutable, powerShellScript, command, exitCode

Set fileSystem = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
powerShellExecutable = fileSystem.BuildPath( _
    shell.ExpandEnvironmentStrings("%SystemRoot%"), _
    "System32\WindowsPowerShell\v1.0\powershell.exe" _
)
powerShellScript = fileSystem.BuildPath(scriptDirectory, "start-network.ps1")
command = """" & powerShellExecutable & """ -NoProfile -NonInteractive -WindowStyle Hidden " & _
    "-ExecutionPolicy Bypass -File """ & powerShellScript & """ -Quiet"

' WScript is a GUI-subsystem host, so window style 0 prevents a console from
' being created before PowerShell can process -WindowStyle Hidden.
exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode
