; Inno Setup script for Talkie-Putty.
; Compile:  iscc /DAppVersion=1.2.3 installer.iss
; Expects PyInstaller output in dist\Talkie-Putty\.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "Talkie-Putty"
#define AppExe "Talkie-Putty.exe"
#define AppPublisher "atornes"
#define AppURL "https://github.com/atornes/Talkie-Putty"

[Setup]
AppId={{8F4B2C1A-9D3E-4A6F-B7C8-1A2B3C4D5E6F}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; Lets the (silent) updater installer detect/close a running instance before
; replacing files. The app creates this same named mutex at startup.
AppMutex=TalkiePutty.SingleInstance
CloseApplications=yes
RestartApplications=no
DisableProgramGroupPage=yes
OutputDir=installer-output
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\{#AppName}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
; No skipifsilent: a silent (auto-update) install relaunches the app too.
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall
