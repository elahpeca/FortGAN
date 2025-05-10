program main
    use gan_arch, only: gan
    use gan_factories, only: create_generator, create_discriminator
    use data_utils, only: load_data
    use gan_train, only: train_gan_step
    use nf, only: network, adam
    implicit none

    type(gan) :: g
    type(network), pointer :: gen, disc
    real, allocatable :: training_images(:,:)
    real, allocatable :: validation_images(:,:)
    real, allocatable :: testing_images(:,:)
    real, allocatable :: params(:)
    integer :: batch_size, noise_dim, epochs
    integer :: i, batch_start, batch_end
    integer :: total_batches, current_batch
    real :: gen_lr = 0.0002, disc_lr = 0.0002  

    noise_dim = 100
    batch_size = 50
    epochs = 10

    call g%init( &
        create_generator(noise_dim=noise_dim, img_size=784), &
        create_discriminator(img_size=784) &
    )

    gen => g%get_generator()
    disc => g%get_discriminator()

    print *, "=== Generator Architecture ==="
    call gen%print_info()
    print *, new_line('a')//"=== Discriminator Architecture ==="
    call disc%print_info()

    call load_data(training_images, validation_images, testing_images, normalize=.true.)
    print *, "Loaded ", size(training_images,2), " training images"
    print *, "Image dimensions: ", size(training_images,1)
    print *, "Batch size: ", batch_size
    print *, "Total batches per epoch: ", ceiling(real(size(training_images,2))/batch_size)

    do i = 1, epochs
        current_batch = 0
        total_batches = ceiling(real(size(training_images,2))/batch_size)
        
        do batch_start = 1, size(training_images, 2), batch_size
            current_batch = current_batch + 1
            batch_end = min(batch_start + batch_size - 1, size(training_images, 2))

            if (mod(current_batch, 10) == 0) then
                print *, "Epoch:", i, "Batch:", current_batch, "/", total_batches

            end if

            call train_gan_step( &
                gen=gen, &
                disc=disc, &
                real_images=training_images(:, batch_start:batch_end), &
                batch_size=batch_size, &
                noise_dim=noise_dim, &
                gen_optimizer=adam(learning_rate=gen_lr, beta1=0.5), &  
                disc_optimizer=adam(learning_rate=disc_lr, beta1=0.5) & 
            )
        end do
        
        print *, "Completed Epoch:", i
        print *, "----------------------------------------"
    end do

    call g%destroy()
    print *, "Training completed successfully."
    
end program main