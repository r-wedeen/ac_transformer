import jax.numpy as jnp
import numpy as np
from jax import jit, lax, debug
from functools import partial

class ACMDP:
    def __init__(self, max_relator_length=1024):
        self.max_relator_length = max_relator_length

        self.action_methods = [
            self._concatenate1, 
            self._concatenate2, 
            lambda s: self._conjugate1(s, 1), 
            lambda s: self._conjugate1(s, -1),
            lambda s: self._conjugate1(s, 2),
            lambda s: self._conjugate1(s, -2),
            lambda s: self._conjugate2(s, 1),
            lambda s: self._conjugate2(s, -1),
            lambda s: self._conjugate2(s, 2),
            lambda s: self._conjugate2(s, -2),
            self._invert1, 
            self._invert2
        ]

        self.inverse_action_methods = [
            self._concatenate1i,
            self._concatenate2i,
            lambda s: self._conjugate1(s, -1), 
            lambda s: self._conjugate1(s, 1),
            lambda s: self._conjugate1(s, -2),
            lambda s: self._conjugate1(s, 2),
            lambda s: self._conjugate2(s, -1),
            lambda s: self._conjugate2(s, 1),
            lambda s: self._conjugate2(s, -2),
            lambda s: self._conjugate2(s, 2),
            self._invert1,
            self._invert2
        ]

        self.n_actions = len(self.action_methods)
    
    # ------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------

    @partial(jit, static_argnames=['self'])
    def transition(self, state, action, inverse=False):
        return lax.cond(
            inverse,
            lambda state: lax.switch(action, self.inverse_action_methods, state),
            lambda state: lax.switch(action, self.action_methods, state),
            operand=state
        ) 
    
    def random_state(self, max_length):
        """returns a random state with relators of length at most max_length"""
        r1 = np.random.choice([-2, -1, 1, 2], size=np.random.randint(1, max_length))
        r2 = np.random.choice([-2, -1, 1, 2], size=np.random.randint(1, max_length))
        r1 = np.pad(r1, (0, self.max_relator_length - len(r1)), mode='constant', constant_values=0)
        r2 = np.pad(r2, (0, self.max_relator_length - len(r2)), mode='constant', constant_values=0)
        s = jnp.array(np.concatenate([r1, r2]))
        return self._reduce(s)
    
    def tuple_to_array(self, state):
        state = tuple((elem,) if isinstance(elem, int) else elem for elem in state)
        assert len(state[0]) <= self.max_relator_length, "r1 exceeds max relator length"
        assert len(state[1]) <= self.max_relator_length, "r2 exceeds max relator length"
        r1 = jnp.pad(jnp.array(state[0]), (0, self.max_relator_length - len(state[0])), mode='constant', constant_values=0)
        r2 = jnp.pad(jnp.array(state[1]), (0, self.max_relator_length - len(state[1])), mode='constant', constant_values=0)
        return jnp.concatenate([r1, r2])
    
    def array_to_tuple(self, state):
        state = np.array(state)
        [r1, r2] = np.split(state, 2)
        r1 = tuple(np.trim_zeros(r1, 'b'))
        r2 = tuple(np.trim_zeros(r2, 'b'))
        return (r1, r2)
    
    # ------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------

    def _concatenate1(self, state):
        lax.cond(
            False, #jnp.sum(state!=0) > self.max_relator_length, 
            lambda: debug.print(f"len(r1+r2) > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r1, r2 = jnp.split(state, 2)
        r1_num_nonzero = jnp.sum(r1 != 0)
        result = jnp.zeros(2*self.max_relator_length, dtype=state.dtype)
        result = result.at[:self.max_relator_length].set(r1)
        result = lax.dynamic_update_slice(result, r2, (r1_num_nonzero,))
        result = result.at[self.max_relator_length:].set(r2)
        return self._reduce(result)

    def _concatenate1i(self, state):
        return self._invert2(self._concatenate1(self._invert2(state)))

    def _concatenate2(self, state):
        lax.cond(
            #jnp.sum(state!=0) > self.max_relator_length, 
            False,
            lambda: debug.print(f"len(r1+r2) > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r1, r2 = jnp.split(state, 2)
        r2_num_nonzero = jnp.sum(r2 != 0)
        temp = jnp.zeros(3*self.max_relator_length, dtype=state.dtype)
        temp = temp.at[0:self.max_relator_length].set(r1)
        temp = temp.at[self.max_relator_length:2*self.max_relator_length].set(r2)
        temp = lax.dynamic_update_slice(temp, r1, (self.max_relator_length + r2_num_nonzero,))
        result = temp[:2*self.max_relator_length]
        return self._reduce(result)

    def _concatenate2i(self, state):
        return self._invert1(self._concatenate2(self._invert1(state)))

    def _conjugate1(self, state, i):
        [r1, r2] = jnp.split(state, 2)
        lax.cond(
            False, #jnp.sum(r1!=0) + 2 > self.max_relator_length, 
            lambda: debug.print(f"len(r1)+2 > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r1_zero_ix = jnp.where(r1 == 0, size=1, fill_value=-1)[0][0]
        r1 = r1.at[r1_zero_ix].set(-i)
        r1 = jnp.roll(r1, 1)
        r1 = r1.at[0].set(i)
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _conjugate2(self, state, i): 
        [r1, r2] = jnp.split(state, 2)
        lax.cond(
            False, #jnp.sum(r2!=0) + 2 > self.max_relator_length, 
            lambda: debug.print(f"len(r2)+2 > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r2_num_nonzero = jnp.sum(r2 != 0)
        r2 = r2.at[r2_num_nonzero].set(-i)
        r2 = jnp.roll(r2, 1)
        r2 = r2.at[0].set(i)
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _invert1(self, state):  
        [r1, r2] = jnp.split(state, 2)
        r1_num_zeros = jnp.sum(r1 == 0)
        r1 = jnp.roll(r1, r1_num_zeros)
        r1 = -r1[::-1]
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _invert2(self, state):
        [r1, r2] = jnp.split(state, 2)
        r2_num_zeros = jnp.sum(r2 == 0)
        r2 = jnp.roll(r2, r2_num_zeros)
        r2 = -r2[::-1]
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _reduce(self, state):
        [r1, r2] = jnp.split(state, 2)
        
        def reduce_relator(r):
            r = jnp.concatenate([r, jnp.array([0, 0])])
            ix = jnp.where(r[:-1] + r[1:] == 0, size=1, fill_value=-1)[0][0]
            r = jnp.roll(r, -ix)
            r = r[2:]
            r = jnp.roll(r, ix)
            return r
        
        def cond_fn(carry):
            r, new_r = carry
            return jnp.any(r != new_r)

        def body_fn(carry):
            _, r = carry
            new_r = reduce_relator(r)
            return r, new_r

        init_val = (jnp.zeros_like(r1), r1)
        (_, r1) = lax.while_loop(cond_fn, body_fn, init_val)
        init_val = (jnp.zeros_like(r2), r2)
        (_, r2) = lax.while_loop(cond_fn, body_fn, init_val)
        return jnp.concatenate((r1, r2))