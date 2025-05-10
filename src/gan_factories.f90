module gan_factories
    use nf, only: network, dense, input, relu, tanhf, sigmoid
    implicit none

    contains

        function create_generator(noise_dim, img_size) result(gen)
            integer, intent(in) :: noise_dim, img_size
            type(network), pointer :: gen

            allocate(gen, source=network([ &
                input(noise_dim), &
                dense(256, activation=relu()), &
                dense(512, activation=relu()), &
                dense(img_size, activation=tanhf()) &
            ]))

        end function

        function create_discriminator(img_size) result(dis)
            integer, intent(in) :: img_size
            type(network), pointer :: dis

            allocate(dis, source=network([ &
                input(img_size), &
                dense(256, activation=relu()), &
                dense(128, activation=relu()), &
                dense(1, activation=sigmoid()) &
            ]))

        end function

end module