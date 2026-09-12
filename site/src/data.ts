/**
 * Fetch a JSON aggregate from `public/data/`.
 *
 * The path goes through BASE_URL so the same code works at `/` in dev and at
 * `/broadsheet/` on GitHub Pages. The files are content-stable between
 * pipeline runs, so `force-cache` keeps a re-render from hitting the network.
 */
export async function loadJson<T>(name: string): Promise<T> {
  const url = `${import.meta.env.BASE_URL}data/${name}`;
  try {
    return await read<T>(url, name, "force-cache");
  } catch {
    // A failure is cacheable too, whether it arrives as a 404 or as a server's
    // HTML fallback, and a cached failure would leave every retry button on
    // the page unable to succeed. One failed read is repeated past the cache.
    return await read<T>(url, name, "reload");
  }
}

async function read<T>(
  url: string,
  name: string,
  cache: RequestCache,
): Promise<T> {
  const res = await fetch(url, { cache });
  if (!res.ok) throw new Error(`Could not load ${name} (${res.status})`);
  return (await res.json()) as T;
}
