const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  getApiBase: () => ipcRenderer.invoke("get-api-base"),
  openFileDialog: () => ipcRenderer.invoke("open-file-dialog"),
  openSampleInputs: () => ipcRenderer.invoke("open-sample-inputs"),
  saveFile: (base64Data, defaultName) =>
    ipcRenderer.invoke("save-file", base64Data, defaultName),
});
