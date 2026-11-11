import io
script = r'''$ErrorActionPreference = "Continue"
$log = "D:\智造创新实践\build\export.log"
function Log($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $m) -Encoding UTF8 }
Set-Content -Path $log -Value "start" -Encoding UTF8
$src = "D:\智造创新实践\build\out.docx"
$pdf = "D:\智造创新实践\build\out.pdf"
if (Test-Path $pdf) { Remove-Item $pdf -Force }
Log "creating word"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
Log "word created"
$doc = $null
try {
  Log "opening"
  $doc = $word.Documents.Open($src, $false, $true, $false)
  Log "opened"
  $doc.Repaginate()
  Log ("pages=" + $doc.ComputeStatistics(2))
  $doc.SaveAs([ref]$pdf, [ref]17)
  Log "saved pdf"
  $doc.Close($false)
  $doc = $null
  Log "closed doc"
} catch {
  Log ("ERROR: " + $_.Exception.Message)
}
if ($null -ne $doc) { try { $doc.Close($false) } catch { Log "close failed" } }
try { $word.Quit() } catch { Log ("quit failed: " + $_.Exception.Message) }
Log "done"
'''
p = r"D:\智造创新实践\build\export.ps1"
with io.open(p, "w", encoding="utf-8-sig", newline="\r\n") as f:
    f.write(script)
print("ok")
