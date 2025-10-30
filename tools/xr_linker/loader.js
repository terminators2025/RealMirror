/**
 * XR Linker - Production Loader
 * 加载编译后的字节码文件
 */
require('bytenode');

// 直接 require 字节码文件
// bytenode 会自动处理 .jsc 文件的加载
require('./src/server/index.jsc');
