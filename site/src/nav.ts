/**
 * The sticky nav says where the reader is. An observer with a thin band near
 * the top of the viewport decides which section is current, so the highlight
 * changes at a line the reader can feel rather than at an arbitrary pixel.
 */
export function initNav(): void {
  const links = new Map<string, HTMLAnchorElement>();
  for (const a of document.querySelectorAll<HTMLAnchorElement>(
    '.section-nav a[href^="#"]',
  )) {
    links.set(a.hash.slice(1), a);
  }
  const sections = [...links.keys()]
    .map((id) => document.getElementById(id))
    .filter((s): s is HTMLElement => s !== null);
  if (!sections.length || typeof IntersectionObserver === "undefined") return;

  const order = sections.map((s) => s.id);
  const visible = new Set<string>();

  const mark = (id: string): void => {
    for (const [key, link] of links) {
      if (key === id) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    }
  };

  const obs = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target.id);
        else visible.delete(entry.target.id);
      }
      // The topmost section in the band wins, so a short section that scrolls
      // past never steals the highlight from the one being read.
      const current = order.find((id) => visible.has(id));
      if (current) mark(current);
    },
    { rootMargin: "-15% 0px -75% 0px", threshold: 0 },
  );
  for (const section of sections) obs.observe(section);
}
