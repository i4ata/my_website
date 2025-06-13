// chatgpt carried here
window.dccFunctions = window.dccFunctions || {};

// Show commas (e.g., 1,000,000)
window.dccFunctions.commaFormat = function(value) {
    return value.toLocaleString();
};

// Show "1.2B", "500M", etc.
window.dccFunctions.abbreviateFormat = function(value) {
    if (value >= 1e9) return (value / 1e9).toFixed(1).replace(/\.0$/, '') + "B";
    if (value >= 1e6) return (value / 1e6).toFixed(1).replace(/\.0$/, '') + "M";
    return value.toLocaleString();
};
