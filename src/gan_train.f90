module gan_train_single
    use gan_arch, only: gan
    use nf_network, only: network
    use nf_optimizers, only: optimizer_base_type
    use nf_random, only: random_normal
    use nf_loss, only: loss_type
    use nf_layer, only: layer
    implicit none

contains

    subroutine train_gan_single( &
        gan_instance, &
        real_image, &
        noise_dim, &
        optimizer_gen, &
        optimizer_disc, &
        loss_disc, &
        loss_gen &
    )
        class(gan), intent(inout) :: gan_instance
        real, intent(in) :: real_image(:)
        integer, intent(in) :: noise_dim
        class(optimizer_base_type), intent(inout) :: optimizer_gen, optimizer_disc
        class(loss_type) :: loss_gen, loss_disc

        type(network), pointer :: gen, disc
        real, allocatable :: real_target(:)
        real, allocatable :: fake_target(:)
        real, allocatable :: noise(:)
        real, allocatable :: fake_image(:)

        gen => gan_instance%get_generator()
        disc => gan_instance%get_discriminator()

        allocate(real_target(1))
        real_target = 1.0
        call train_disc_single(disc, real_image, real_target, optimizer_disc, loss_disc)
        deallocate(real_target)

        allocate(noise(noise_dim))
        call random_normal(noise)
        call gen%forward(noise)
        fake_image = gen%predict(noise)
        allocate(fake_target(1))
        fake_target = 0.0
        call train_disc_single(disc, fake_image, fake_target, optimizer_disc, loss_disc)
        deallocate(noise)
        deallocate(fake_image)
        deallocate(fake_target)

        allocate(noise(noise_dim))
        call random_normal(noise)
        call gen%forward(noise)
        fake_image = gen%predict(noise)
        call disc%forward(fake_image)
        allocate(real_target(1))
        real_target = 1.0
        call train_gen_single(gen, disc, fake_image, real_target, optimizer_gen, loss_gen)
        deallocate(noise)
        deallocate(real_target)

    end subroutine train_gan_single

    subroutine train_disc_single(disc, data, target, optimizer, loss)
        class(network), intent(inout) :: disc
        real, intent(in) :: data(:)
        real, intent(in) :: target(:)
        class(optimizer_base_type), intent(inout) :: optimizer
        class(loss_type), intent(in) :: loss

        real, allocatable :: output(:)
        real :: loss_val

        call disc%forward(data)
        output = disc%predict(data)
        print *, output
        print *, target
        loss_val = loss%eval(target, output)
        print *, "Discriminator loss (single):", loss_val

        call disc%backward(target, loss)
        call disc%update(optimizer, 1)

    end subroutine train_disc_single

subroutine train_gen_single(gen, disc, fake_data, target, optimizer, loss)
    class(network), intent(inout) :: gen
    class(network), intent(inout) :: disc
    real, intent(in) :: fake_data(:)
    real, intent(in) :: target(:) ! Цель (1.0)
    class(optimizer_base_type), intent(inout) :: optimizer
    class(loss_type) :: loss

    real, allocatable :: disc_output(:)
    type(layer), allocatable :: a
    real :: loss_val

    !!! impossible without specific realization of GAN backbrop =(

end subroutine train_gen_single

end module gan_train_single