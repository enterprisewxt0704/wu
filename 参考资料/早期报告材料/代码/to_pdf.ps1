$ErrorActionPreference = "Stop"
$src = "D:\智造创新实践\从一只瓶盖看懂智能制造_四讲知识点整理.docx"
$pdf = "D:\智造创新实践\preview.pdf"
if (Test-Path $pdf) { Remove-Item $pdf -Force }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($src, $false, $true)
  $doc.Repaginate()
  $pages = $doc.ComputeStatistics(2)   # wdStatisticPages
  Write-Output "PAGES=$pages"
  $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 0, $true, $true, 0, $true, $true, $false)
  Write-Output "PDF=$pdf"
  $doc.Close($false)
} finally {
  $word.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
