cmake_minimum_required(VERSION 3.9)
project(Nicu C CXX)
# Console window: a copy of Retro68's RetroConsole with a larger font (see console/).
add_application(Nicu nicu.c nicu.r
    console/Console.cc console/ConsoleWindow.cc console/InitConsole.cc)
set_target_properties(Nicu PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(Nicu PRIVATE console)
target_compile_definitions(Nicu PRIVATE MAC_TARGET)
target_compile_options(Nicu PRIVATE -O2 -ffast-math -mcpu=7400 -maltivec -mabi=altivec -mvrsave -Wa,-maltivec -fno-tree-vectorize -fno-tree-slp-vectorize)
target_link_libraries(Nicu m)
