const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("api", {
  getApiBase: () => ipcRenderer.invoke("get-api-base"),
  openFileDialog: () => ipcRenderer.invoke("open-file-dialog"),
  readDroppedFile: (filePath) => ipcRenderer.invoke("read-dropped-file", filePath),
  getPathForFile: (file) => webUtils.getPathForFile(file),
  openSampleInputs: () => ipcRenderer.invoke("open-sample-inputs"),
  saveFile: (base64Data, defaultName) =>
    ipcRenderer.invoke("save-file", base64Data, defaultName),
});
