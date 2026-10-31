param([Parameter(Mandatory=$true)][string]$InputDocx, [Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference='Stop'
$word=$null
$document=$null
try {
    $word=New-Object -ComObject Word.Application
    $word.Visible=$false
    $word.DisplayAlerts=0
    $word.AutomationSecurity=3
    $document=$word.Documents.Open([IO.Path]::GetFullPath($InputDocx),$false,$true)
    $document.Repaginate()
    $document.ExportAsFixedFormat([IO.Path]::GetFullPath($OutputPdf),17)
    Write-Output ('Rendered pages: '+$document.ComputeStatistics(2))
} finally {
    if($document){$document.Close(0);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)}
    if($word){$word.Quit();[void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)}
}
