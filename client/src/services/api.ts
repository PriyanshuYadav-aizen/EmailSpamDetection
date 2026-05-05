import axios from "axios";

const isLocalhost =
  typeof window !== "undefined" &&
  ["localhost", "127.0.0.1"].includes(window.location.hostname);

const api = axios.create({
  baseURL: isLocalhost
    ? "http://localhost:5000/api"
    : import.meta.env.VITE_API_URL || "https://spam-node-server.onrender.com/api",
});

export default api;