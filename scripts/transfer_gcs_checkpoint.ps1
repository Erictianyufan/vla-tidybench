#!/usr/bin/env pwsh

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Bucket,

    [Parameter(Mandatory = $true)]
    [string]$Prefix,

    [Parameter(Mandatory = $true)]
    [string]$Remote,

    [Parameter(Mandatory = $true)]
    [string]$RemoteRoot,

    [Parameter(Mandatory = $true)]
    [string]$IdentityFile,

    [string]$StagingDirectory = (Join-Path $env:TEMP "vla-tidybench-gcs-transfer")
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-Md5Hex {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -Algorithm MD5 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Convert-Base64Md5ToHex {
    param([Parameter(Mandatory = $true)][string]$Value)

    return -join ([Convert]::FromBase64String($Value) | ForEach-Object { $_.ToString("x2") })
}

$normalizedPrefix = $Prefix.TrimStart("/")
if (-not $normalizedPrefix.EndsWith("/")) {
    $normalizedPrefix += "/"
}
$normalizedRemoteRoot = $RemoteRoot.TrimEnd("/")
$manifestPath = Join-Path $StagingDirectory "objects.json"
New-Item -ItemType Directory -Force -Path $StagingDirectory | Out-Null

$encodedPrefix = [Uri]::EscapeDataString($normalizedPrefix)
$manifestUrl = "https://storage.googleapis.com/storage/v1/b/$Bucket/o?prefix=$encodedPrefix&maxResults=1000"
Invoke-Checked -Description "GCS manifest download" -Command {
    curl.exe --ssl-no-revoke --fail --location --retry 8 --retry-all-errors `
        --silent --show-error --output $manifestPath $manifestUrl
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.nextPageToken) {
    throw "The object listing is paginated; narrow the prefix or add pagination support."
}
$objects = @($manifest.items | Where-Object { $_.name -and -not $_.name.EndsWith("/") })
if ($objects.Count -eq 0) {
    throw "No objects found below gs://$Bucket/$normalizedPrefix"
}

$totalBytes = ($objects | ForEach-Object { [int64]$_.size } | Measure-Object -Sum).Sum
Write-Host "Transferring $($objects.Count) objects ($totalBytes bytes)"

$index = 0
foreach ($object in $objects) {
    $index += 1
    $relativePath = $object.name.Substring($normalizedPrefix.Length)
    $localPath = Join-Path $StagingDirectory $relativePath
    $localParent = Split-Path -Parent $localPath
    $remotePath = "$normalizedRemoteRoot/$relativePath"
    $remoteParent = $remotePath.Substring(0, $remotePath.LastIndexOf("/"))
    $expectedSize = [int64]$object.size
    $expectedMd5 = Convert-Base64Md5ToHex $object.md5Hash
    New-Item -ItemType Directory -Force -Path $localParent | Out-Null

    Write-Host "[$index/$($objects.Count)] $relativePath ($expectedSize bytes)"
    $needsDownload = -not (Test-Path -LiteralPath $localPath)
    if (-not $needsDownload) {
        $needsDownload = (Get-Item -LiteralPath $localPath).Length -ne $expectedSize -or `
            (Get-Md5Hex $localPath) -ne $expectedMd5
    }
    if ($needsDownload) {
        $objectUrl = "https://storage.googleapis.com/$Bucket/$($object.name)"
        Invoke-Checked -Description "Download of $relativePath" -Command {
            curl.exe --ssl-no-revoke --fail --location --retry 8 --retry-all-errors `
                --connect-timeout 30 --continue-at - --output $localPath $objectUrl
        }
    }

    $actualSize = (Get-Item -LiteralPath $localPath).Length
    $actualMd5 = Get-Md5Hex $localPath
    if ($actualSize -ne $expectedSize -or $actualMd5 -ne $expectedMd5) {
        throw "Local verification failed for $relativePath"
    }

    Invoke-Checked -Description "Remote directory creation" -Command {
        ssh -i $IdentityFile -o BatchMode=yes -o IdentitiesOnly=yes $Remote `
            "mkdir -p -- '$remoteParent'"
    }
    Invoke-Checked -Description "Upload of $relativePath" -Command {
        scp -i $IdentityFile -o BatchMode=yes -o IdentitiesOnly=yes `
            $localPath "${Remote}:$remotePath"
    }
    $remoteDigest = ssh -i $IdentityFile -o BatchMode=yes -o IdentitiesOnly=yes $Remote `
        "test \"`$(stat -c %s '$remotePath')\" = '$expectedSize' && md5sum '$remotePath' | cut -d' ' -f1"
    if ($LASTEXITCODE -ne 0 -or $remoteDigest.Trim() -ne $expectedMd5) {
        throw "Remote verification failed for $relativePath"
    }

    Remove-Item -LiteralPath $localPath
    Write-Host "[$index/$($objects.Count)] verified on $Remote"
}

Write-Host "Transfer complete: ${Remote}:$normalizedRemoteRoot"
