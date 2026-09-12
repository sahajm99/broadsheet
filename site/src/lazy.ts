/** Run `cb` once, when `el` is near the viewport. */
export function whenVisible(
  el: Element,
  cb: () => void,
  rootMargin = "200px",
): void {
  if (typeof IntersectionObserver === "undefined") {
    cb();
    return;
  }
  const obs = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        obs.disconnect();
        cb();
        return;
      }
    },
    { rootMargin },
  );
  obs.observe(el);
}
