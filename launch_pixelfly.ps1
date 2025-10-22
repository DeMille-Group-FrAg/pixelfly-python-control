# Set the AppUserModelID before launching Python
$appid = 'PixelFly.PythonControl'
[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')
$script = @"
using System;
using System.Runtime.InteropServices;

public class TaskbarHelper {
    [DllImport("shell32.dll", SetLastError = true)]
    public static extern void SetCurrentProcessExplicitAppUserModelID([MarshalAs(UnmanagedType.LPWStr)] string AppID);
}
"@
Add-Type -TypeDefinition $script
[TaskbarHelper]::SetCurrentProcessExplicitAppUserModelID($appid)

# Run Python
cd 'c:\Users\13128\jmd\pixelfly-python-control\'
& 'C:\ProgramData\Miniconda3\python.exe' 'main.py'