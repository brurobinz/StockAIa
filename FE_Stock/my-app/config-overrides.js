const webpack = require("webpack");

module.exports = function override(config, env) {
    config.resolve.fallback = {
        ...config.resolve.fallback,
        http: require.resolve("stream-http"),
        https: require.resolve("https-browserify"),
        stream: require.resolve("stream-browserify"),
        zlib: require.resolve("browserify-zlib"),
        buffer: require.resolve("buffer"),
        util: require.resolve("util/"),
        assert: require.resolve("assert/"),
        url: require.resolve("url/"),
        process: require.resolve("process/browser"), // Ensure this is included
    };
    config.plugins = (config.plugins || []).concat([
        new webpack.ProvidePlugin({
            process: "process/browser", // Ensure this is provided globally
            Buffer: ["buffer", "Buffer"],
        }),
    ]);
    return config;
};
