import tensorflow as tf
from keras.utils import losses_utils
import matplotlib.pyplot as plt
from typing import Tuple
from waveoptics.metrics.tf import pearson, quality



def scaled_sigmoid(x: tf.Tensor, amp: float, coeff: float, offset: float,
                       dynamic: float = 1, dynamic_scaling: bool = True,
                       invert: bool = False,) -> tf.Tensor:
    sign = -1 if invert else 1
    if dynamic_scaling:
        x = x / dynamic
    arg = sign * coeff * (x - offset)
    return amp * tf.nn.sigmoid(arg)




def is_complex_dtype(x: tf.Tensor) -> bool:
  complex_dtypes = [tf.dtypes.complex64, tf.dtypes.complex128]
  return True if x.dtype in complex_dtypes else False



def get_tf_shape(inputs: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    shape = tf.shape(inputs)
    return tuple(shape[i] for i in range(len(shape)))


class _BaseCustomLoss(tf.keras.losses.Loss):
    """
        Base custom loss class to inherit from.
        Embeds complex data type conversion, arguments preparation, and metric inversion.
    """
    def __init__(self,
                 inversed: bool = True, # Wether metric is returned 
                 squared: bool = False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None,):
        """
            Inputs:
            - inversed [bool]: Loss function returns 1 - metric if True
            - squared [bool]: Input field is squared if True
            - reduction: default metric reduction if loss is multidimensional
            - name [str]: name
        """
        super().__init__(reduction, name)
        self.inversed_metric: bool = inversed
        self.squared_args: bool = squared

    def _invert_metric(self, x: float) -> float:
        return 1 - x if self.inversed_metric else x
    
    def _prepare_args(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor]:
        """
            Prepare arguments for loss function.
            - Returns a abs(x) if x is a complex tensor
            - Returns square(x) if squared option has been selected
        """
        x = tf.math.abs(x) if is_complex_dtype(x) else x
        y = tf.math.abs(y) if is_complex_dtype(y) else y
        if self.squared_args:
            x = tf.math.square(x)
            y = tf.math.square(y)
        return x, y
    
    def _loss(self, x: tf.Tensor, y: tf.Tensor) -> float:
        "To be defined in subclasses. Must return a real value."
        pass



class MAE(_BaseCustomLoss):
    """
        Computes the mean absolute error between the groundtruth and the prediction
    """
    def __init__(self,
                 inversed: bool = False,
                 squared: bool = False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='MAE',):
        super().__init__(inversed, squared, reduction, name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        y_true, y_pred = self._prepare_args(y_true, y_pred)
        return self._loss(y_true, y_pred)
    
    def _loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        metric = tf.reduce_mean(tf.abs(y_true - y_pred))
        return self._invert_metric(metric)



class MSE(_BaseCustomLoss):
    """
        Computes the mean square error between the groundtruth and the prediction
    """
    def __init__(self,
                 inversed: bool = False,
                 squared: bool = False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='MSE',):
        super().__init__(inversed, squared, reduction, name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        y_true, y_pred = self._prepare_args(y_true, y_pred)
        return self._loss(y_true, y_pred)
    
    def _loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        metric = tf.reduce_mean(tf.square(y_true - y_pred))
        return self._invert_metric(metric)
    

class PearsonBatch(_BaseCustomLoss):
    """
        Computes the Pearson correlation coefficient between the groundtruth and the prediction
    """

    def __init__(self,
                 inversed: bool = True,
                 squared: bool = False,
                 axis: list[int] = None,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='PCC'):
        super().__init__(inversed, squared, reduction, name)
        self.axis = axis

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        y_true, y_pred = self._prepare_args(y_true, y_pred)
        return self._loss(y_pred, y_true)
        # return self._invert_metric(self._loss(y_pred, y_true))
    
    def _loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        s = tf.math.reduce_sum((y_true - tf.math.reduce_mean(y_true)) * (y_pred - tf.math.reduce_mean(y_pred)) / tf.cast(tf.size(y_true), tf.float32))
        p = s / (tf.math.reduce_std(y_true) * tf.math.reduce_std(y_pred))
        return self._invert_metric(p)
    

class Pearson(_BaseCustomLoss):
    """
        Computes the Pearson correlation coefficient between the groundtruth and the prediction
    """

    def __init__(self,
                 inversed: bool = True,
                 squared: bool = False,
                 axis: list[int] = None,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='PCC'):
        super().__init__(inversed, squared, reduction, name)
        self.axis = axis

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        y_true, y_pred = self._prepare_args(y_true, y_pred)
        return self._loss(y_pred, y_true)
        # return self._invert_metric(self._loss(y_pred, y_true))
    
    def _loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        redux_ytrue = y_true - tf.math.reduce_mean(y_true, axis=(1, 2), keepdims=True)
        redux_ypred = y_pred - tf.math.reduce_mean(y_pred, axis=(1, 2), keepdims=True)
        s = tf.math.reduce_sum(redux_ytrue * redux_ypred / tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), y_true.dtype), axis=(1, 2), keepdims=True)
        p = s / (tf.math.reduce_std(y_true, axis=(1, 2), keepdims=True) * tf.math.reduce_std(y_pred, axis=(1, 2), keepdims=True))
        p = tf.reduce_mean(p)
        return self._invert_metric(p)









class SSIM(_BaseCustomLoss):
    """
        Computes the Structural Similarity between the groundtruth and the prediction
    """
    def __init__(self,
                 inversed: bool = True,
                 squared: bool = False,
                 max_val: float = 1.0,
                 filter_size: int = 11,
                 reduction=losses_utils.ReductionV2.AUTO, 
                 name='SSIM',):
        super().__init__(inversed, squared, reduction, name)
        self.max_val = max_val
        self.filter_size = filter_size

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        y_true, y_pred = self._prepare_args(y_true, y_pred)
        return self._loss(y_true, y_pred)
    
    def _loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        if len(y_pred.shape) < 4:
            y_true, y_pred = tf.expand_dims(y_true, -1), tf.expand_dims(y_pred, -1)
        s = tf.image.ssim(y_pred, y_true, max_val = self.max_val, filter_size = self.filter_size)
        return self._invert_metric(s)
    

class SSIM_mod(_BaseCustomLoss):
    """
        Computes the Structural Similarity between the groundtruth and the prediction
    """
    def __init__(self,
                 inversed: bool = True,
                 squared: bool = False,
                 max_val: float = 1.0,
                 filter_size: int = 11,
                 reduction=losses_utils.ReductionV2.AUTO, 
                 name='SSIM',):
        super().__init__(inversed, squared, reduction, name)
        self.max_val = max_val
        self.filter_size = filter_size

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        y_true, y_pred = self._prepare_args(y_true, y_pred)
        return self._loss(y_true, y_pred)
    
    def _loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        if len(y_pred.shape) < 3:
            y_true, y_pred = tf.expand_dims(y_true, -1), tf.expand_dims(y_pred, -1)
        s = tf.image.ssim(y_pred, y_true, max_val = self.max_val, filter_size = self.filter_size)
        return self._invert_metric(s)



    





class _BaseComplexCustomLoss(_BaseCustomLoss):
    """
        Base custom loss class to inherit from, used for complex predictions.
        Embeds complex data type conversion, arguments preparation, and metric inversion.
    """
    def __init__(self, pad_to: int, inversed: bool = True, squared: bool = False, loss_product: bool = False, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(inversed, squared, reduction, name)
        self._fourier_energy_ratio: float = None
        self._fourier_energy_ratio2: float = None
        self._mmf_energy_ratio: float = 1
        self._mmf_energy_ratio2: float = 1
        self.energy_reg: bool = False
        self.energy_reg_weight: float = None
        self._loss_product: bool = loss_product
        self.pad_to = pad_to
        self._init_shape = None
    
    def _prepare_args(self, y_true, y_pred):
        y_true, z_true = self._split_groundtruth_channels(y_true)
        y_true = tf.reshape(y_true, (-1, 64, 64))
        z_true = tf.reshape(z_true, (-1, 64, 64))
        y_pred = tf.reshape(y_pred, (-1, 64, 64))
        z_pred = self._make_fourier_prediction(y_pred)
        return y_true, z_true, y_pred, z_pred


    def _pad_to_size(self, tensor):
        if self._init_shape is None:
            self._init_shape = tensor.shape
            self._pad_amount = (self.pad_to - self._init_shape[1]) // 2
        tensor = tf.pad(tensor, [[0, 0], [self._pad_amount, self._pad_amount], [self._pad_amount, self._pad_amount]])
        return tensor
        
    def _split_groundtruth_channels(self, x: tf.Tensor) -> tuple[tf.Tensor]:
        return x[..., 0], x[..., 1]

    def _make_complex_prediction(self, x: tf.Tensor) -> tf.Tensor:
        return tf.complex(x[..., 0], x[..., 1])
    
    def _make_fourier_prediction(self, y_pred_cplx: tf.Tensor) -> tf.Tensor:
        init_shape = y_pred_cplx.shape
        pad_amount = (self.pad_to - init_shape[1]) // 2

        total_batch_energy = self.energy(y_pred_cplx)
        y_pred_cplx = tf.pad(y_pred_cplx, [[0, 0], [pad_amount, pad_amount], [pad_amount, pad_amount]])
        # y_pred_cplx = tf.pad(y_pred_cplx, [[0, 0], [init_shape[1]//2, init_shape[1]//2], [init_shape[2]//2, init_shape[2]//2]])
        y_pred_cplx = self.fft2(y_pred_cplx, normalize=True)
        final_shape = y_pred_cplx.shape

        total_batch_energy = self.energy(y_pred_cplx)
        crop = (final_shape[1] - init_shape[1]) // 2
        y_pred_cplx = y_pred_cplx[:, crop:-crop, crop:-crop]
        total_batch_cropped_fft2_energy = self.energy(y_pred_cplx)
        self._fourier_energy_ratio = total_batch_cropped_fft2_energy / total_batch_energy
        return y_pred_cplx
    
    def _make_new_fourier_prediction(self, y_pred_cplx: tf.Tensor) -> tf.Tensor:
        z_pred_cplx = self.fft2(y_pred_cplx, normalize=True)
        self._fourier_energy_ratio = self.energy(z_pred_cplx) / self.energy(y_pred_cplx)
        return z_pred_cplx
    
    def _batch_energy_ratio(self, groundtruth: tf.Tensor, prediction: tf.Tensor, threshold: float = 0.01) -> float:
        # Compute the average intensity from the groundtruth batch and normalize it by its maximum
        avg_int_gt = tf.reduce_mean(tf.square(tf.math.abs(groundtruth[..., 0])), axis=0)
        avg_int_gt /= tf.reduce_max(avg_int_gt)

        # Compute the average intensity from the prediction batch and normalize it by its maximum
        avg_int_pr = tf.reduce_mean(tf.square(tf.math.abs(prediction)), axis=0)

        # Compute energies and return ratio
        pr_energy =  tf.reduce_sum(avg_int_pr)
        pr_energy_within_gt_trsh = tf.reduce_sum(tf.where(tf.math.greater_equal(avg_int_gt, threshold), avg_int_pr, 0))
        return pr_energy_within_gt_trsh / pr_energy
    
    def _batch_energy_ratio2(self, groundtruth: tf.Tensor, prediction: tf.Tensor) -> float:
        # Compute energies and return ratio
        pr_energy = tf.reduce_sum(tf.square(tf.math.abs(prediction)))
        gt_energy = tf.reduce_sum(tf.square(tf.math.abs(groundtruth)))
        return 0.1 * tf.math.abs((pr_energy - gt_energy) / gt_energy)
    
    @staticmethod
    def fft2(field: tf.Tensor, normalize: bool = True) -> tf.Tensor:
        ft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(field)))
        if normalize:
            field_shape = field.shape
            numel_image = tf.math.reduce_prod(field_shape[-2:])
            # if tf.size(field_shape) >= 3:
            #     numel_image = tf.math.reduce_prod(field_shape[-2:])
            # else:
            #     numel_image = tf.math.reduce_prod(field_shape)
            return ft / tf.cast(tf.math.sqrt(tf.cast(numel_image, tf.float64)), ft.dtype)
        else:
            return ft
    
    @staticmethod
    def energy(field: tf.Tensor) -> float:
        return tf.reduce_sum(tf.math.square(tf.math.abs(field)))
    
    def _energy_regularized_loss(self, coumpound_loss: float) -> float:
        if self.energy_reg:
            reg_loss_ft = 1 * self.energy_reg_weight * self._invert_metric(self._fourier_energy_ratio)
            reg_loss_ft2 = 1 * self.energy_reg_weight * self._invert_metric(self._fourier_energy_ratio2)
            reg_loss_sp = 0 * self.energy_reg_weight * self._invert_metric(self._mmf_energy_ratio)
            reg_loss_sp2 = 0 * self.energy_reg_weight * self._invert_metric(self._mmf_energy_ratio2)
            if self._loss_product:
                return coumpound_loss * reg_loss_ft * reg_loss_ft2 * reg_loss_sp
            else:
                return coumpound_loss + reg_loss_ft + reg_loss_ft2 + reg_loss_sp
        else:
            return coumpound_loss
        
    def kl_loss_reim_ypred(self, y_pred_cplx):
        real, imag = tf.math.real(y_pred_cplx), tf.math.imag(y_pred_cplx)
        real = (real - tf.reduce_mean(real)) / (3 * tf.math.reduce_std(real))
        imag = (imag - tf.reduce_mean(imag)) / (3 * tf.math.reduce_std(imag))
        kl_loss = tf.keras.losses.KLDivergence()
        kl_loss1 = kl_loss(tf.math.abs(real), tf.math.abs(imag))
        kl_loss2 = kl_loss(tf.math.abs(imag), tf.math.abs(real))
        # kl_loss1 = kl_loss(real + 0.5, imag + 0.5)
        # kl_loss2 = kl_loss(imag + 0.5, real + 0.5)
        return 1 * tf.maximum(kl_loss1, 0) + 0 * tf.maximum(kl_loss2, 0)
    










class SSIM_MMF_and_Pearson_Fourier(_BaseComplexCustomLoss):
    """
        Computes the SSIM of the amplitude/intensity at the fiber output,
        and the PCC of the amplitude/intensity at the Fourier plane.
    """
    def __init__(self,
                 pad_to: int,
                 weights: tuple[float] = (0.5, 0.5),
                 inversed: bool = True,
                 squared: bool = False,
                 max_val: float = 1.0,
                 filter_size: int = 11,
                 energy_regularization: bool = False,
                 energy_reg_weight: float = 0.5,
                 loss_product: bool = False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name="SSIM_MMF_PCC_FT",):
        super().__init__(pad_to, inversed, squared, loss_product, reduction, name)
        self.loss_object1 = SSIM(inversed=False, squared=squared, max_val=max_val, filter_size=filter_size)
        self.loss_object2 = Pearson(inversed=False, squared=squared)
        self.weights = weights
        self.energy_reg = energy_regularization
        self.energy_reg_weight = energy_reg_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true, z_true, y_pred, z_pred = self._prepare_args(y_true, y_pred)
        self._mmf_energy_ratio = self._batch_energy_ratio(y_true, y_pred)
        self._mmf_energy_ratio2 = self._batch_energy_ratio2(y_true, y_pred)
        self._fourier_energy_ratio2 = self._batch_energy_ratio(z_true, z_pred)
        metric = self._coumpound_loss(y_true, y_pred, z_true, z_pred)
        kld_loss = self.kl_loss_reim_ypred(y_pred)
        metric = metric + 0 * 1e-2 * kld_loss
        return self._energy_regularized_loss(metric)
    
    def _loss1(self, x: tf.Tensor, y: tf.Tensor) -> float:
        return self._invert_metric(self.loss_object1(x, y))
    
    def _loss2(self, x: tf.Tensor, y: tf.Tensor) -> float:
        return self._invert_metric(self.loss_object2(x, y))
        
    def _coumpound_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, z_true: tf.Tensor, z_pred: tf.Tensor) -> tf.Tensor:
        loss_nf = self.weights[0] * self._loss1(y_true, y_pred)
        loss_ff = self.weights[1] * self._loss2(z_true, z_pred)
        return loss_nf * loss_ff if self._loss_product else loss_nf + loss_ff



class Dynamic_Pearson_MMF_and_SSIM_Fourier_v3(SSIM_MMF_and_Pearson_Fourier):
    """
        Computes the PCC of the amplitude/intensity at the fiber output,
        and the SSIM of the amplitude/intensity at the Fourier plane.
    """
    def __init__(self,
                 pad_to: int,
                 weights: tuple[float] = (0.5, 0.5),
                 inversed: bool = True,
                 squared: bool = False,
                 max_val: float = 1,
                 filter_size: int = 11,
                 energy_regularization: bool = False,
                 energy_reg_weight: float = 0.5,
                 loss_product: bool = False,
                 scaled_sigmoid_settings: dict = dict(amp=0.5, coeff=30, offset=0.01, dynamic=1, invert=True),
                 reduction=losses_utils.ReductionV2.AUTO,
                 name="PCC_MMF_SSIM_FT",):
        super().__init__(pad_to, weights, inversed, squared, max_val, filter_size, energy_regularization, energy_reg_weight, loss_product, reduction, name)
        self.scaled_sigmoid_settings = scaled_sigmoid_settings
        self.loss_object1 = Pearson(inversed=False, squared=squared)
        self.loss_object2 = SSIM(inversed=False, squared=squared, max_val=max_val, filter_size=filter_size)

    def _coumpound_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, z_true: tf.Tensor, z_pred: tf.Tensor) -> tf.Tensor:
        loss1 = self._loss1(y_true, y_pred)
        loss2 = self._loss2(z_true, z_pred)
        loss_nf = loss1
        loss_ff = loss2 * scaled_sigmoid(loss1, **self.scaled_sigmoid_settings)
        return loss_nf * loss_ff if self._loss_product else loss_nf + loss_ff
    
    def plot_scaled_sigmoid(self):
        x = tf.experimental.numpy.arange(0, 1, 0.001)
        w = scaled_sigmoid(x, **self.scaled_sigmoid_settings)
        plt.plot(x, w)




