# The 9B that was scored is the 9B that ships, apart from its license file (2026-10-23)

What this shows: the 9B numbers in [`results/v1.json`](../v1.json) were measured on a build named
`grux-local-9b-mo` on the 16 GB Mac mini (Apple M4, Ollama 0.40.8). The published
`gruxai.com/grux/grux-local:9b` is that build with one layer swapped: its license file, replaced by the license blob
that `:2b` and `:4b` already ship. The other 768 layers and the config are the same bytes, so the scored numbers are
the published model's numbers.

The two manifests, compared on October 23, 2026 at 6:35 PM Eastern:

- scored: `grux-local-9b-mo`, manifest `sha256:d32ed272f3c307911c3a417a3b6aec996ae8160bcd5f7be69b8373917299e136`
  (157,958 bytes), from a copy of its manifest blob saved at 5:06 PM; that name has since been removed from the
  Mac mini;
- published: `grux-local:9b` on the same Mac mini, manifest
  `sha256:357e4bd45ff0030a6dbfdf5a7650b745418f1dcdb7eeed473b4078987f111837` (157,958 bytes), the digest in
  `results/v1.json`.

The comparison hashes each file first, and both hashes match their names. Its output, verbatim:

```
scored    manifest sha256:d32ed272f3c307911c3a417a3b6aec996ae8160bcd5f7be69b8373917299e136
published manifest sha256:357e4bd45ff0030a6dbfdf5a7650b745418f1dcdb7eeed473b4078987f111837
layers: 769 vs 769
config identical: True sha256:72125e423a5127807c9b82c5bd5dfcaccb34181bcc9c6e2a7480b061e9aab4cb 220
layers that differ: 1
  layer 767: mediaType application/vnd.ollama.image.license -> application/vnd.ollama.image.license
    scored    sha256:5153ce5bc5e5a4e0151f878690119e49ee8953aeecf994a48615f0754e1da73b 11344 bytes
    published sha256:c553d8e3621054d3c6bf0e5f385a5ef72353609a52142e78dc8e544d7e357703 11545 bytes
layers identical (digest, size, mediaType, order): 768
non-layer keys identical: {'schemaVersion': True, 'mediaType': True, 'config': True, 'runner': True, 'format': True}
scored-license.txt: sha256:5153ce5bc5e5, 11344 bytes, CRLF 0, bare LF 202
shipped-license.txt: sha256:c553d8e36210, 11545 bytes, CRLF 201, bare LF 1
license text identical once CRLF is read as LF: True
```

The one layer that differs is the license file. Both versions carry the same Apache License 2.0 text: the scored
build has LF line endings and the shipped one has CRLF. The shipped blob, `sha256:c553d8e3...` (11,545 bytes), is
also the license layer of `grux-local:2b` (manifest `sha256:601cd23c66ef...`) and `grux-local:4b` (manifest
`sha256:1de6472396f2...`), as read from their manifests on the same Mac mini.
