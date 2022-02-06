

function(copy_kernels)
    set(kernels
            NormalizeCharToFloat.cl
            )
    foreach (kernel ${kernels})
        configure_file(
                ${kernel}
                ${CMAKE_BINARY_DIR}/bin/kernels/${kernel}
                COPYONLY
        )
    endforeach ()
endfunction()

copy_kernels()