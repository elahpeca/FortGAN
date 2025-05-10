program main
    use gan_arch, only: gan
    use gan_factories, only: create_generator, create_discriminator
    use nf, only: network
    implicit none
  
    type(gan) :: g
    type(network), pointer :: gen, disc
    call g%init( &
        create_generator(100, 784), &
        create_discriminator(784) &
    )

    gen => g%get_generator()
    disc => g%get_discriminator()

    write(*, "(A)") "Generator info:"
    write(*, *)
    call gen%print_info()

    write(*, "(A)") "Discriminator info:"
    write(*, *)
    call disc%print_info()

    call g%destroy()

end program main