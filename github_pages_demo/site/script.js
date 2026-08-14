const metrics = {
  accuracy: "97.44%",
  f1: "0.974",
  latency: "~120 ms",
  modelSize: "~98 MB"
};

document.getElementById("metric-acc").textContent = metrics.accuracy;
document.getElementById("metric-f1").textContent = metrics.f1;
document.getElementById("metric-lat").textContent = metrics.latency;
document.getElementById("metric-size").textContent = metrics.modelSize;

document.getElementById("year").textContent = `Updated ${new Date().getFullYear()}`;
