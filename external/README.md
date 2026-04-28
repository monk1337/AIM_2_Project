# external/ third-party model repositories

This study evaluates and fine-tunes four hand-pose models hosted in their own
upstream repositories. Rather than vendoring the upstream code, we pin exact
commits and apply small patches where needed.

## Pinned upstream commits

| Repo            | Upstream URL                                       | Commit    | Notes                            |
| --------------- | -------------------------------------------------- | --------- | -------------------------------- |
| WiLoR           | https://github.com/rolpotamias/WiLoR.git           | `fcb9113` | clean checkout                   |
| HandOccNet      | https://github.com/namepllet/HandOccNet.git        | `65ba997` | clean checkout                   |
| MeshGraphormer  | https://github.com/microsoft/MeshGraphormer.git    | `27f7cdb` | clean checkout                   |
| HaMeR           | https://github.com/geopavlakos/hamer.git           | `3a01849` | apply `PATCHES/hamer.patch`      |
| HOISDF          | https://github.com/amathislab/HOISDF.git           | `666e5b7` | clean checkout                   |
| InterWild       | https://github.com/facebookresearch/InterWild.git  | `7c01e4a` | clean checkout                   |
| hands           | https://github.com/ap229997/hands.git              | `8303277` | clean checkout                   |

## Setting them up as submodules

```bash
cd external
for repo in WiLoR HandOccNet MeshGraphormer hamer HOISDF InterWild hands; do
  case "$repo" in
    WiLoR)          url=https://github.com/rolpotamias/WiLoR.git          sha=fcb9113 ;;
    HandOccNet)     url=https://github.com/namepllet/HandOccNet.git       sha=65ba997 ;;
    MeshGraphormer) url=https://github.com/microsoft/MeshGraphormer.git   sha=27f7cdb ;;
    hamer)          url=https://github.com/geopavlakos/hamer.git          sha=3a01849 ;;
    HOISDF)         url=https://github.com/amathislab/HOISDF.git          sha=666e5b7 ;;
    InterWild)      url=https://github.com/facebookresearch/InterWild.git sha=7c01e4a ;;
    hands)          url=https://github.com/ap229997/hands.git             sha=8303277 ;;
  esac
  git submodule add "$url" "$repo"
  ( cd "$repo" && git checkout "$sha" )
done

# apply our hamer patch
( cd hamer && git apply ../PATCHES/hamer.patch )
```

## What `hamer.patch` changes

50 lines. Two files:
* `hamer/models/hamer.py`: 4 lines changed; small forward-pass tweak.
* `hamer/utils/__init__.py`: 21 lines changed; bf16 / device-handling cleanup.

Apply with `git apply external/PATCHES/hamer.patch` from inside `external/hamer/`.

## Why submodules instead of vendoring

* Avoids re-distributing third-party code under their own licenses.
* Pins exact commits so reviewers can reproduce.
* Keeps this repo's history small (the seven repos sum to roughly 800 MB).
