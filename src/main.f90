program main
    use gan_arch, only: gan
    use gan_factories, only: create_generator, create_discriminator
    use data_utils, only: load_data
    use nf, only: network
    implicit none
  
    type(gan) :: g
    type(network), pointer :: gen, disc
    real, allocatable :: training_images(:,:)
    real, allocatable :: validation_images(:,:)
    real, allocatable :: testing_images(:,:)

    call g%init( &
        create_generator(noise_dim=100, img_size=784), &
        create_discriminator(img_size=784) &
    )

    gen => g%get_generator()
    disc => g%get_discriminator()

    write(*, "(A)") "Generator info:"
    write(*, *) 
    call gen%print_info()

    write(*, "(A)") "Discriminator info:"
    write(*, *)
    call disc%print_info()

    call load_data(training_images, validation_images, testing_images, normalize=.true.)
    
    call g%destroy()
  
end program main