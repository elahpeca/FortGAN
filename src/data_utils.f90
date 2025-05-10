module data_utils
    use nf, only: load_mnist
    implicit none

    contains

        subroutine load_data(training_images, validation_images, testing_images, normalize)
            real, allocatable, intent(out) :: training_images(:,:)
            real, allocatable, intent(out) :: validation_images(:,:)
            real, allocatable, intent(out) :: testing_images(:,:)
            real, allocatable :: training_labels(:), validation_labels(:), testing_labels(:)
            logical, intent(in) :: normalize

            call load_mnist( &
                training_images=training_images, &
                validation_images=validation_images, &
                testing_images=testing_images, &
                training_labels=training_labels, &
                validation_labels=validation_labels, &
                testing_labels=testing_labels  &
            )

            if (normalize) then
                call normalize_images(training_images)
                call normalize_images(validation_images)
                call normalize_images(testing_images)
            end if

        end subroutine

        subroutine normalize_images(images)
            real, intent(inout) :: images(:,:)
            images = 2 * images - 1

        end subroutine

end module