// Nod service worker: the app works offline once loaded (except the gateway, of course).
//   App shell (HTML, JS, CSS): network first, cache fallback, so updates arrive at once.
//   API (/api/, /healthz): network first, cache fallback.
//   Pinned libraries and models (vendor/, cdn.jsdelivr.net, storage.googleapis.com): cache first.
// WebSockets never pass through a service worker.

const VERSION = 'nod-v0.3.0';
const SHELL = [
  './', 'index.html', 'presenter.html', 'manifest.webmanifest', 'css/app.css', 'css/presenter.css',
  'icons/icon.svg', 'icons/icon-192.png', 'icons/icon-512.png',
  'js/app.js', 'js/presenter.js', 'js/gestures.js', 'js/calibrate.js', 'js/ui.js', 'js/net.js', 'js/estop.js',
  'js/proto.js', 'js/keyboard.js', 'js/face.js', 'js/audio.js', 'js/kws.js', 'js/tts.js',
];
const IMMUTABLE_HOSTS = ['cdn.jsdelivr.net', 'storage.googleapis.com'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('nod-') && k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

async function networkFirst(req) {
  const cache = await caches.open(VERSION);
  try {
    const res = await fetch(req);
    if (res.ok && req.method === 'GET') cache.put(req, res.clone());
    return res;
  } catch (e) {
    const hit = await cache.match(req, { ignoreSearch: req.mode === 'navigate' });
    if (hit) return hit;
    throw e;
  }
}

async function cacheFirst(req) {
  const cache = await caches.open(VERSION);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok || res.type === 'opaque') cache.put(req, res.clone());
  return res;
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin === self.location.origin) {
    if (url.pathname.includes('/vendor/')) { event.respondWith(cacheFirst(req)); return; }
    event.respondWith(networkFirst(req));
    return;
  }
  if (IMMUTABLE_HOSTS.includes(url.hostname)) event.respondWith(cacheFirst(req));
});
