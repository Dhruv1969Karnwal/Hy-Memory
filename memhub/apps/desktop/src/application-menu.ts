import type { Menu, MenuItemConstructorOptions } from "electron";

export function installApplicationMenu(options: {
  productName: string;
  buildMenu(template: MenuItemConstructorOptions[]): Menu;
  setMenu(menu: Menu): void;
  exportLogs(): Promise<void>;
  reportError(message: string): void;
}): void {
  const runExport = async () => {
    try {
      await options.exportLogs();
    } catch {
      options.reportError("The log export failed; try again shortly.");
    }
  };
  options.setMenu(
    options.buildMenu([
      {
        label: options.productName,
        submenu: [{ role: "about" }, { type: "separator" }, { role: "quit" }],
      },
      { role: "editMenu" },
      { role: "windowMenu" },
      {
        label: "Help",
        submenu: [
          {
            label: "Export logs (send to developers for debugging)",
            click: runExport,
          },
        ],
      },
    ]),
  );
}
