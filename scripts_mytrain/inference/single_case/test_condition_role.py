import unittest,torch
from condition_role_case import condition_input
class ConditionTest(unittest.TestCase):
 def test_roles_preserve_latents_and_future_time(self):
  x=torch.randn(1,4,13,2,2)
  for b in ['wan','nano','predict25']:
   if b=='wan':kw=dict(hidden_states=x,timestep=torch.full((1,52),995.))
   elif b=='nano':kw=dict(vision_tokens=[x],vision_token_shapes=[(13,2,2)],vision_timesteps=torch.full((52,),995.),vision_mse_loss_indexes=['preserved'])
   else:kw=dict(x_B_C_T_H_W=x,timesteps_B_T=torch.full((1,13),995.),condition_video_input_mask_B_C_T_H_W=torch.zeros(1,1,13,2,2))
   saved=x.clone();out,actual,role=condition_input(b,kw,.1)
   self.assertIs(actual,x);self.assertTrue(torch.equal(x,saved));self.assertEqual(role['future_time'],995.)
   if b=='wan':self.assertTrue((out['timestep'][:,:8]==0).all())
   elif b=='nano':
    self.assertEqual(out['vision_timesteps'].numel(),44);self.assertEqual(out['vision_noisy_frame_indexes'][0].tolist(),list(range(2,13)))
    self.assertIs(out['vision_mse_loss_indexes'],kw['vision_mse_loss_indexes'])
    out2,_,_=condition_input(b,out,.1);self.assertTrue(torch.equal(out2['vision_timesteps'],out['vision_timesteps']))
   else:
    self.assertTrue((out['condition_video_input_mask_B_C_T_H_W'][:,:,:2]==1).all())
    self.assertTrue((out['condition_video_input_mask_B_C_T_H_W'][:,:,2:]==0).all())
    self.assertTrue((out['timesteps_B_T'][:,:2]==.1).all())
if __name__=='__main__':unittest.main()
