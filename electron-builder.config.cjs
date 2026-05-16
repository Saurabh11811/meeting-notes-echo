const macSigningEnabled = process.env.ECHO_MAC_SIGN === "1";

/** @type {import("electron-builder").Configuration} */
module.exports = {
  appId: "com.humanlyst.echo",
  productName: "ECHO",
  artifactName: "${productName}-${version}-${os}-${arch}.${ext}",
  directories: {
    output: "release",
  },
  files: [
    "electron/**/*",
    "ui/dist/**/*",
    "package.json",
  ],
  extraResources: [
    {
      from: "backend-dist",
      to: "backend",
      filter: ["**/*"],
    },
  ],
  mac: {
    category: "public.app-category.productivity",
    identity: macSigningEnabled ? undefined : "-",
    hardenedRuntime: macSigningEnabled,
    gatekeeperAssess: false,
    target: ["dmg", "zip"],
  },
  dmg: {
    sign: macSigningEnabled,
  },
  win: {
    target: ["nsis", "msi", "portable"],
  },
  linux: {
    category: "Office",
    target: ["AppImage", "deb"],
  },
  nsis: {
    oneClick: false,
    perMachine: false,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
  },
};
