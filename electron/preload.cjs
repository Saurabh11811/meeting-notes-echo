const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("echoDesktop", {
  platform: process.platform,
  openExternal: (url) => ipcRenderer.invoke("echo:openExternal", url),
});
