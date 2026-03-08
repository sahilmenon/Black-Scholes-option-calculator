export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const targetPath = "/project/streamlitapp";
    const streamlitUpstream = "https://black-scholes-option-calculator-sahil.streamlit.app/";

    // Check if the request is for the streamlit subdirectory
    if (url.pathname.startsWith(targetPath)) {
      // Reconstruct the URL for the Streamlit Cloud app
      const remainingPath = url.pathname.replace(targetPath, "");
      const newUrl = new URL(streamlitUpstream + remainingPath + url.search);
      
      // Clone the request but change the URL
      const modifiedRequest = new Request(newUrl, request);
      
      // Fetch from Streamlit and return to user
      return fetch(modifiedRequest);
    }

    // Otherwise, let the request pass through to your main website (Pages)
    return fetch(request);
  }
}