param([string]$src, [string]$log)
function Log($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss.fff), $m) -Encoding UTF8 }
Set-Content -Path $log -Value ("--- " + $src) -Encoding UTF8
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
Log "created"
$doc = $null
try {
  $doc = $word.Documents.Open($src, $false, $true, $false)
  Log "opened"
  $doc.Repaginate()
  Log ("pages=" + $doc.ComputeStatistics(2))
  Log ("paras=" + $doc.Paragraphs.Count)
  Log ("tables=" + $doc.Tables.Count)
  $doc.Close($false)
  $doc = $null
  Log "closed"
} catch {
  Log ("ERROR: " + $_.Exception.Message)
}
if ($null -ne $doc) { try { $doc.Close($false) } catch {} }
try { $word.Quit() } catch {}
Log "done"
