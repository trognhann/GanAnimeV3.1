
$ErrorActionPreference = "Stop"
$baseDir = Get-Location
Write-Output "BaseDir: $baseDir" | Out-File debug.log -Encoding UTF8

function Get-EscapedContent($paths) {
    $fullContent = ""
    foreach ($path in $paths) {
        $realPath = Join-Path $baseDir $path
        
        if (Test-Path $realPath) {
            Write-Output "Reading $realPath" | Out-File debug.log -Append -Encoding UTF8
            $c = Get-Content $realPath -Raw -Encoding UTF8
            
            # Simple clean imports by line
            # robust split
            $lines = $c -split "\r?\n"
            $filtered = @()
            foreach ($line in $lines) {
                if ($line -notmatch "^\s*from tools" -and $line -notmatch "^\s*from net" -and $line -notmatch "^\s*from AnimeGAN") {
                    $filtered += $line
                }
            }
            $c = $filtered -join "`n"
            $fullContent += $c + "`n"
        } else {
            Write-Output "File not found: $realPath" | Out-File debug.log -Append -Encoding UTF8
        }
    }
    
    # Escape for JSON string
    $escaped = $fullContent -replace "\\", "\\"
    $escaped = $escaped -replace '"', '\"'
    $escaped = $escaped -replace "`n", "\n"
    $escaped = $escaped -replace "`r", ""
    
    # Escape $ for Regex replacement string in PowerShell
    $escaped = $escaped -replace '\$', '$$$$'
    
    return $escaped
}

$jsonPath = Join-Path $baseDir "train.ipynb"
if (-not (Test-Path $jsonPath)) {
    Write-Output "JSON not found: $jsonPath" | Out-File debug.log -Append
    exit
}

$json = Get-Content $jsonPath -Raw -Encoding UTF8

# 1. TF Color Ops
$code = Get-EscapedContent @("tools\tf_color_ops.py")
$json = $json -replace "# TF_COLOR_OPS_PLACEHOLDER", $code

# 2. VGG19
$code = Get-EscapedContent @("tools\vgg19.py")
$json = $json -replace "# VGG19_PLACEHOLDER", $code

# 3. Ops
$code = Get-EscapedContent @("tools\ops.py")
$json = $json -replace "# OPS_PLACEHOLDER", $code

# 4. Other Utils
$code = Get-EscapedContent @("tools\utils.py", "tools\common_utils.py", "tools\GuidedFilter.py", "tools\L0_smoothing.py", "tools\data_loader.py")
$json = $json -replace "# OTHER_UTILS_PLACEHOLDER", $code

# 5. Network
$code = Get-EscapedContent @("net\generator.py", "net\discriminator.py")
$json = $json -replace "# NETWORK_PLACEHOLDER", $code

# 6. AnimeGAN
$code = Get-EscapedContent @("AnimeGANv3_shinkai.py")
$json = $json -replace "# ANIMEGAN_PLACEHOLDER", $code

# 7. Train
$code = Get-EscapedContent @("train.py")
# Comment out train call
$code = $code -replace "if __name__", "# if __name__" -replace "train\(\)", "# train()"
$json = $json -replace "# TRAIN_PLACEHOLDER", $code

# 8. Test
$code = Get-EscapedContent @("test.py")
# Comment out test call
$code = $code -replace "if __name__", "# if __name__" -replace "test\(", "# test("
$json = $json -replace "# TEST_PLACEHOLDER", $code

# 9. Example
# Just static string replacement for example
$json = $json -replace "# EXAMPLE_PLACEHOLDER", "# Run training:\n# parse_args = lambda: TrainConfig()\n# train()"

$json | Set-Content $jsonPath -Encoding UTF8
Write-Output "Done building." | Out-File debug.log -Append
