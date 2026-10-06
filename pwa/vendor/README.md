# vendor/

Pinned browser libraries and models for offline use, downloaded by `sh scripts/vendor.sh`
(hash-checked). The files themselves are not committed; only this README is.
`js/face.js` and `js/kws.js` read `vendor/manifest.json` and fall back to the CDN without it.
