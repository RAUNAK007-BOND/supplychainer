// Supplychainer app shell cache: opens instantly from the home screen, even before the server wakes.
// API calls and the live WebSocket are never cached.
const CACHE = 'sc-shell-v1';
const SHELL = ['/app/', '/app/manifest.webmanifest', '/app/icons/icon-192.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/api/') || url.pathname === '/ws') return;

  // Hashed build files never change: cache first.
  if (url.pathname.startsWith('/assets/')) {
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    })));
    return;
  }

  // The app page itself: serve the cached copy at once and refresh it in the background,
  // so the next launch picks up a new deploy.
  if (req.mode === 'navigate' && url.pathname.startsWith('/app')) {
    e.respondWith(caches.open(CACHE).then(async (c) => {
      const cached = await c.match('/app/');
      const fresh = fetch(req).then((res) => { if (res.ok) c.put('/app/', res.clone()); return res; }).catch(() => cached);
      return cached || fresh;
    }));
  }
});
