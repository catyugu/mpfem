function(mpfem_set_default_compiler_options target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_20)

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive- /utf-8 /wd4819 /bigobj)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  if(MINGW)
    add_compile_options(-Wa,-mbig-obj)
  endif()
endfunction()
