import { el } from "../figure.ts";

/**
 * The stand-in every figure container gets until its own chart task lands.
 * It fills the height the real figure will occupy (`--fig-h` on the
 * container), so the page does not jump when the chart arrives, and it says
 * plainly that nothing failed.
 */
export async function render(container: HTMLElement): Promise<void> {
  container.replaceChildren();
  const box = el(
    "div",
    "fig-skeleton",
    "Chart arrives in a later task",
  );
  container.appendChild(box);
}
