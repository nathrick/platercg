cmake_minimum_required(VERSION 3.8)

project(platercg VERSION 0.0.1 DESCRIPTION "License plate recognition lib")
include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" ON)

# Link this 'library' to use the following warnings
add_library(project_warnings INTERFACE)

if(CMAKE_COMPILER_IS_GNUCC)
  option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" FALSE)

  if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0)
  endif()
endif()

if(MSVC)
  target_compile_options(project_warnings INTERFACE /W4)
else()
  target_compile_options(project_warnings
                         INTERFACE
                         -Wall
                         -Wextra # reasonable and standard
                         -Wshadow # warn the user if a variable declaration shadows one from a
                                  # parent context
                         -Wnon-virtual-dtor # warn the user if a class with virtual functions has a
                                            # non-virtual destructor. This helps catch hard to
                                            # track down memory errors
                         -Wold-style-cast # warn for c-style casts
                         -Wcast-align # warn for potential performance problem casts
                         -Wunused # warn on anything being unused
                         -Woverloaded-virtual # warn if you overload (not override) a virtual
                                              # function
                         -Wpedantic # warn if non-standard C++ is used
                         -Wconversion # warn on type conversions that may lose data
                         -Wsign-conversion # warn on sign conversions
                         -Wmisleading-indentation # warn if identation implies blocks where blocks
                                                  # do not exist
                         -Wduplicated-cond # warn if if / else chain has duplicated conditions
                         -Wduplicated-branches # warn if if / else branches have duplicated code
                         -Wlogical-op # warn about logical operations being used where bitwise were
                                      # probably wanted
                         -Wnull-dereference # warn if a null dereference is detected
                         -Wuseless-cast # warn if you perform a cast to the same type
                         -Wdouble-promotion # warn if float is implicit promoted to double
                         -Wformat=2 # warn on security issues around functions that format output
                                    # (ie printf)
                         )
endif()

set(SOURCE_FILES
    ${PROJECT_NAME}/character.cpp
    ${PROJECT_NAME}/knn_rcg.cpp
    )


set(HEADER_FILES
    include/${PROJECT_NAME}/platercg.hpp
    include/${PROJECT_NAME}/knn_rcg.hpp
    )

add_library(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

find_package(OpenCV REQUIRED)

set_target_properties(${PROJECT_NAME} PROPERTIES
    			VERSION ${PROJECT_VERSION}
    			SOVERSION 1
                        PUBLIC_HEADER include/${PROJECT_NAME}/platercg.hpp)

configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} project_warnings --coverage)
target_include_directories(${PROJECT_NAME} PRIVATE .)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Config
    	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
    	DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

