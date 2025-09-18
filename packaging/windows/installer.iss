#define MyAppVersion GetDefine("MyAppVersion", "0.0.0")
#define Sign GetDefine("Sign", "0")
#define PfxPath GetEnv("N2D_CODESIGN_PFX_PATH")
#define PfxPassword GetEnv("N2D_CODESIGN_PFX_PASSWORD")
#define TimestampUrl GetEnv("N2D_CODESIGN_TIMESTAMP_URL")

[Setup]
AppId={{D7F9B58E-2E8C-4D2D-8A08-29E5EB7A1AF4}}
AppName=normal2disp GUI
AppVersion={#MyAppVersion}
AppPublisher=<Your Org>
DefaultDirName={autopf}\normal2disp GUI
DefaultGroupName=normal2disp GUI
UninstallDisplayIcon={app}\normal2disp-gui.exe
OutputBaseFilename=normal2disp-GUI-{#MyAppVersion}
OutputDir=dist\windows\{#MyAppVersion}\installer
SetupIconFile=assets\icons\normal2disp.ico
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64
DisableDirPage=no
DisableProgramGroupPage=no

#if Sign == "1"
SignedUninstaller=yes
SignTool=SignTool
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\\windows\\{#MyAppVersion}\\bundle\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\normal2disp GUI"; Filename: "{app}\\normal2disp-gui.exe"
Name: "{commondesktop}\\normal2disp GUI"; Filename: "{app}\\normal2disp-gui.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\\normal2disp-gui.exe"; Description: "{cm:LaunchProgram,normal2disp GUI}"; Flags: nowait postinstall skipifsilent

#if (Sign == "1") && (PfxPath != "") && (PfxPassword != "")
[SignTool]
#if TimestampUrl != ""
Name: "SignTool"; Command: "signtool sign /fd SHA256 /f \"{#PfxPath}\" /p \"{#PfxPassword}\" /tr \"{#TimestampUrl}\" /td SHA256 \"{#file}\""
#else
Name: "SignTool"; Command: "signtool sign /fd SHA256 /f \"{#PfxPath}\" /p \"{#PfxPassword}\" \"{#file}\""
#endif
#endif
