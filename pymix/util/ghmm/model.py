#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/model.c
#*       Authors:  Benhard Knab, Bernd Wichern, Benjamin Georgi, Alexander Schliep
#*
#*       Copyright (C) 1998-2004 Alexander Schliep
#*       Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
#*       Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik,
#*                               Berlin
#*
#*       Contact: schliep@ghmm.org
#*
#*       This library is free software you can redistribute it and/or
#*       modify it under the terms of the GNU Library General Public
#*       License as published by the Free Software Foundation either
#*       version 2 of the License, or (at your option) any later version.
#*
#*       This library is distributed in the hope that it will be useful,
#*       but WITHOUT ANY WARRANTY without even the implied warranty of
#*       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#*       Library General Public License for more details.
#*
#*       You should have received a copy of the GNU Library General Public
#*       License along with this library if not, write to the Free
#*       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#*
#*
#*       This file is version $Revision: 2304 $
#*                       from $Date: 2013-05-31 13:48:13 -0400 (Fri, 31 May 2013) $
#*             last change by $Author: ejb177 $.
#*
#******************************************************************************
from util.ghmm.types import kDiscreteHMM, kBackgroundDistributions, kSilentStates, kNoBackgroundDistribution, kTiedEmissions, kUntied, kHigherOrderEmissions, kLabeledStates
from util.ghmm.wrapper import ARRAY_CALLOC, ARRAY_MALLOC, ARRAY_REALLOC, GHMM_EPS_PREC
from vendor.pyLibrary.env.logs import Log

DONE=0
NOTVISITED=1
VISITED=2




def model_state_alloc(s, M, in_states, out_states):
# define CUR_PROC "model_state_alloc"
  s.b=ARRAY_CALLOC( M)
  if out_states > 0:
    s.out_id=ARRAY_CALLOC( out_states)
    s.out_a=ARRAY_CALLOC( out_states)

  if in_states > 0:
    s.in_id=ARRAY_CALLOC( in_states)
    s.in_a=ARRAY_CALLOC( in_states)


def ghmm_dmodel_calloc(M, N, modeltype, inDegVec, outDegVec) :
  assert(modeltype & kDiscreteHMM)

  mo=ARRAY_CALLOC( 1)

  mo.M = M
  mo.N = N
  mo.model_type = modeltype

  mo.s=ARRAY_CALLOC( N)
  for i in range(N):
      mo.s[i]= model_state_alloc(M, inDegVec[i], outDegVec[i])


  if mo.model_type & kSilentStates:
    mo.silent=ARRAY_CALLOC( N)

  if mo.model_type & kTiedEmissions:
    mo.tied_to=ARRAY_CALLOC( N)
    for i in range(N):
      mo.tied_to[i] = kUntied


  if mo.model_type & kBackgroundDistributions:
    mo.background_id=ARRAY_MALLOC( N)
    for i in range(N):
      mo.background_id[i] = kNoBackgroundDistribution

  if mo.model_type & kHigherOrderEmissions:
    mo.order=ARRAY_CALLOC( N)

  if mo.model_type & kLabeledStates:
    mo.label=ARRAY_CALLOC( N)

  return mo



def model_copy_vectors(mo, index, a_matrix, b_matrix, pi, fix):
  cnt_out = 0
  cnt_in = 0
  mo.s[index].pi = pi[index]
  mo.s[index].fix = fix[index]
  for i in range( 0,  mo.M):
    mo.s[index].b[i] = b_matrix[index][i]
  for i in range( 0,  mo.N):
    if a_matrix[index][i]:   # Transitions to a following state possible
      if cnt_out >= mo.s[index].out_states:
        Log.error("")

      mo.s[index].out_id[cnt_out] = i
      mo.s[index].out_a[cnt_out] = a_matrix[index][i]
      cnt_out+=1

    if a_matrix[i][index]:   # Transitions to a previous state possible
      if cnt_in >= mo.s[index].in_states:
        Log.error("")

      mo.s[index].in_id[cnt_in] = i
      mo.s[index].in_a[cnt_in] = a_matrix[i][index]
      cnt_in+=1

def ghmm_dmodel_read(filename):
  new_models = 0
  mo_number = 0
  s = ighmm_scanner_alloc (filename)

  while not s.err and not s.eof:
    ighmm_scanner_get_name (s)
    ighmm_scanner_consume (s, '=')
    if s.err:
        Log.error(s.err)
    if s.id == "HMM") or s.id == "hmm"):
      (mo_number)+=1
      # more mem
      mo=ARRAY_REALLOC (mo, mo_number)
      mo[mo_number - 1] = ghmm_dmodel_direct_read (s, (int *) &new_models)
      if not mo[mo_number - 1]:
        Log.error("")

      # Copies the model, that has already been read.
      if new_models > 1:
        # "-1" because mo_number++ has already been done.
        mo = ARRAY_REALLOC (mo, mo_number - 1 + new_models)
        for j in range( 1,  new_models):
          mo[mo_number] = ghmm_dmodel_copy (mo[mo_number - 1])
          if not mo[mo_number]:
            Log.error("")

          (mo_number)+=1
    elif s.id == "HMM_SEQ"):
      ghmm_dmodel **tmp_mo = None
      tmp_mo = ghmm_dmodel_from_sequence_ascii (s, &new_models)
      mo = ARRAY_REALLOC (mo, (mo_number + new_models))
      for j in range( 0,  new_models):
        if not tmp_mo[j]:
          Log.error("")

        mo[mo_number] = tmp_mo[j]
        (mo_number)+=1


    else:
      Log.error("")

    ighmm_scanner_consume (s, '')
    if s.err:
      Log.error("")
                               # while not s.err and not s.eof:
  return mo, mo_number


def ghmm_dmodel_direct_read(s, multip):
#define CUR_PROC "ghmm_dmodel_direct_read"
  mt_read=0
  multip = 1                  # default
  mo=ARRAY_CALLOC( 1)

  ighmm_scanner_consume (s, ':')

  while not s.err and not s.eof and s.c - '':
    ighmm_scanner_get_name(s)
    if s.id!="M" and s.id!="N" and s.id!="Pi"     and s.id!="A" and s.id!="B" and s.id!="multip"        and s.id!="prior" and s.id!="fix_state" and s.id!="ModelType" :
      Log.error()

    ighmm_scanner_consume (s, '=')
    if s.err:
      Log.error()
    if s.id=="multip":
      multip = ighmm_scanner_get_int (s)
      if multip < 1:        #
        Log.error("Doesn't make any sense!")


    elif s.id=="M":    #Number of output values
      if m_read:
        Log.error("identifier M twice")

      mo.M = ighmm_scanner_get_int (s)
      m_read = 1

    elif s.id=="N":
      if n_read:
        Log.error("identifier N twice")

      mo.N = ighmm_scanner_get_int (s)
      mo.s=ARRAY_CALLOC( mo.N)
      n_read = 1

    elif s.id=="A":    #Transition probability
      if not n_read:
        Log.error("need N as a range for A")

      if a_read:
        Log.error("identifier A twice")

      a_matrix=ARRAY_CALLOC( mo.N)
      ighmm_scanner_get_name (s)
      if s.id == "matrix"):
        if ighmm_cmatrix_read(s, a_matrix, mo.N, mo.N):
          Log.error("unable to read matrix A")

        a_read = 1

      else:
        Log.error("unknown identifier")

    elif s.id=="B":    #Output probability
      if (not n_read) or (not m_read):
        Log.error("need M and N as a range for B")

      if b_read:
        Log.error("identifier B twice")

      b_matrix=ARRAY_CALLOC( mo.N)
      ighmm_scanner_get_name (s)
      if s.id=="matrix":
        ighmm_cmatrix_read(s, b_matrix, mo.N, mo.M)
        b_read = 1

      else:
        Log.error("unknown identifier")

    elif s.id=="prior":        #A prior model
      if prior_read:
        Log.error("identifier prior twice")

      mo.prior = ighmm_scanner_get_edouble (s)
      if (mo.prior < 0 or mo.prior > 1) and mo.prior != -1:
        Log.error("invalid model prior")
      prior_read = 1

    elif s.id=="ModelType" :    # Model type
      if mt_read:
        Log.error("identifier ModelType twice")
      mo.model_type = ighmm_scanner_get_int(s)
      if mo.model_type & (kLabeledStates + kHigherOrderEmissions):
        Log.error("unsupported Model Type")
      mt_read = 1

    elif s.id == "Pi":
     #Initial state probabilty
      if not n_read:
        Log.error("need N as a range for Pi")
      if pi_read:
        Log.error("identifier Pi twice")

      ighmm_scanner_get_name (s)
      if s.id=="vector":
        ighmm_scanner_consume (s, ':')
        pi_vector = scanner_get_double_earray (s, &len)
        if len != mo.N:
          Log.error("wrong number of elements in PI")

        ighmm_scanner_consume (s, '')
        ighmm_scanner_consume (s, '')
        pi_read = 1

      else:
        Log.error("unknown identifier")

    elif s.id == "fix_state":
      if not n_read:
        Log.error("need N as a range for fix_state")

      if fix_read:
        Log.error("identifier fix_state twice")

      ighmm_scanner_get_name (s)
      if s.id == "vector":
        ighmm_scanner_consume (s, ':')
        fix_vector = scanner_get_int_array (s, &len)
        if len != mo.N:
          Log.error("wrong number of elements in fix_state")
        ighmm_scanner_consume (s, '')
        ighmm_scanner_consume (s, '')
        fix_read = 1
      else:
        Log.error("unknown identifier")
    else:
      Log.error("unknown identifier")
    ighmm_scanner_consume (s, '')
                               # while ..s.c-'':
  ighmm_scanner_consume (s, '')

  # No prior read -. give it the value -1
  if prior_read == 0:
    mo.prior = -1
  # Allocate memory for the model
  for i in range( 0,  mo.N):
    mo.s[i].out_states = ighmm_cmatrix_notzero_columns (a_matrix, i, mo.N)
    mo.s[i].in_states = ighmm_cmatrix_notzero_rows (a_matrix, i, mo.N)
    if (model_state_alloc (mo.s + i, mo.M, mo.s[i].in_states,  mo.s[i].out_states)) :
      Log.error()

    # Assign the parameters to the model
    if not a_matrix:
      Log.error("no A matrix specified in filenot \n")
      exit (1)

    if not b_matrix:
      Log.error("no B matrix specified in filenot \n")
      exit (1)

    if not fix_vector:
      Log.error("no fix_state vector specified in filenot \n")
      exit (1)

    if not pi_vector:
      Log.error("no Pi vector specified in filenot \n")
      exit (1)


    if model_copy_vectors(mo, i, a_matrix, b_matrix, pi_vector, fix_vector):
        Log.error()


#============================================================================
# Produces models from given sequences
def ghmm_dmodel_from_sequence(sq):
  mo=ARRAY_CALLOC( sq.seq_number)
  max_symb = ghmm_dseq_max_symbol (sq)
  for i in range( 0,  sq.seq_number):
    mo[i] = ghmm_dmodel_generate_from_sequence (sq.seq[i], sq.seq_len[i],                                          max_symb + 1)
  return mo


#============================================================================
# Produces models form given sequences
def ghmm_dmodel_from_sequence_ascii(s, mo_number):
  ighmm_scanner_consume (s, ':')
  while not s.err and not s.eof and s.c - '':
    ighmm_scanner_get_name (s)
    ighmm_scanner_consume (s, '=')
    # Reads sequences on normal format
    if s.id == "SEQ"):
      sq = ghmm_dseq_read_alloc (s)
      if not sq:
        Log.error()
    else:
      Log.error("unknown identifier")

    ighmm_scanner_consume (s, '')
  ighmm_scanner_consume (s, '')
  mo=ARRAY_CALLOC( sq.seq_number)
  # The biggest symbol that occurs
  max_symb = ghmm_dseq_max_symbol (sq)
  for i in range( 0,  sq.seq_number):
    mo[i] = ghmm_dmodel_generate_from_sequence (sq.seq[i], sq.seq_len[i],                                       max_symb + 1)

  mo_number = sq.seq_number
  ghmm_dseq_free (&sq)
  return mo



def ghmm_alphabet_free(a) :
  pass


#============================================================================
def ghmm_dmodel_free(mo) :
  pass


#============================================================================
def ghmm_dbackground_free(bg) :
    pass


#============================================================================
def ghmm_dmodel_copy(mo):
  m2=ARRAY_CALLOC( 1)
  m2.s=ARRAY_CALLOC( mo.N)

  if mo.model_type & kSilentStates:
    m2.silent=ARRAY_CALLOC( mo.N)
  if mo.model_type & kTiedEmissions:
    m2.tied_to=ARRAY_CALLOC( mo.N)
  if mo.model_type & kBackgroundDistributions:
    m2.background_id=ARRAY_CALLOC( mo.N)
    m2.bp = mo.bp

  if mo.model_type & kHigherOrderEmissions:
    m2.order=ARRAY_CALLOC( mo.N)
  if mo.model_type & kLabeledStates:
    m2.label=ARRAY_CALLOC( mo.N)

  m2.pow_lookup=ARRAY_MALLOC( mo.maxorder+2)

  for i in range( 0,  mo.N):
    if mo.model_type & kHigherOrderEmissions:
      size = pow( mo.M, mo.order[i]+1)
    else:
      size = mo.M

    nachf = mo.s[i].out_states
    vorg = mo.s[i].in_states

    m2.s[i].out_id=ARRAY_CALLOC( nachf)
    m2.s[i].out_a=ARRAY_CALLOC( nachf)
    m2.s[i].in_id=ARRAY_CALLOC( vorg)
    m2.s[i].in_a=ARRAY_CALLOC( vorg)
    m2.s[i].b=ARRAY_CALLOC( size)

    # copy the values
    for j in range( 0,  nachf):
      m2.s[i].out_a[j] = mo.s[i].out_a[j]
      m2.s[i].out_id[j] = mo.s[i].out_id[j]

    for j in range( 0,  vorg):
      m2.s[i].in_a[j] = mo.s[i].in_a[j]
      m2.s[i].in_id[j] = mo.s[i].in_id[j]

    # copy all b values for higher order states
    for m in range( 0,  size):
      m2.s[i].b[m] = mo.s[i].b[m]

    m2.s[i].pi = mo.s[i].pi
    m2.s[i].fix = mo.s[i].fix
    if mo.model_type & kSilentStates:
      m2.silent[i] = mo.silent[i]
    if mo.model_type & kTiedEmissions:
      m2.tied_to[i] = mo.tied_to[i]
    if mo.model_type & kLabeledStates:
      m2.label[i] = mo.label[i]
    if mo.model_type & kHigherOrderEmissions:
      m2.order[i] = mo.order[i]
    if mo.model_type & kBackgroundDistributions:
      m2.background_id[i] = mo.background_id[i]
    m2.s[i].out_states = nachf
    m2.s[i].in_states = vorg


  m2.N = mo.N
  m2.M = mo.M
  m2.prior = mo.prior
  if mo.model_type & kHigherOrderEmissions:
    m2.maxorder = mo.maxorder
    for i in range( mo.maxorder+2):
      m2.pow_lookup[i] = mo.pow_lookup[i]


  m2.model_type = mo.model_type
  # not necessary but the history is at least initialised
  m2.emission_history = mo.emission_history
  return (m2)

#============================================================================
def ghmm_dmodel_check(mo) :
  imag=0

  # The sum of the Pi[i]'s is 1
  sum = 0.0
  for i in range( 0,  mo.N):
    sum += mo.s[i].pi

  if abs(sum-1.0) >= GHMM_EPS_PREC:
    Log.error("sum Pi[i] != 1.0")

  # check each state
  for i in range(mo.N):
    sum = 0.0
    # Sum the a[i][j]'s : normalized out transitions
    for j in range( mo.s[i].out_states):
      sum += mo.s[i].out_a[j]

    if j==0:
      GHMM_LOG_PRINTF(LDEBUG, LOC, "out_states = 0 (state %d . final statenot )", i)

    elif sum == 0.0:
      GHMM_LOG_PRINTF(LWARN, LOC, "sum of s[%d].out_a[*] = 0.0 (assumed final "
                      "state but %d transitions)", i, mo.s[i].out_states)

    if abs(sum-1.0) >= GHMM_EPS_PREC:
      Log.error("sum of s[%d].out_a[*] = %f != 1.0", i, sum)

    # Sum the a[i][j]'s : normalized in transitions
    sum = mo.s[i].pi
    for j in range(mo.s[i].in_states):
      sum += mo.s[i].in_a[j]

    if abs(sum) <= GHMM_EPS_PREC:
      imag = 1
      Log.error("state %d can't be reached", i)


    # Sum the b[j]'s: normalized emission probs
    sum = 0.0
    for j in range(mo.M):
      sum += mo.s[i].b[j]

    if imag:
      # not reachable states
      if (abs(sum + mo.M ) >= GHMM_EPS_PREC):
        Log.error("state %d can't be reached but is not set as non-reachale state", i)

     elif (mo.model_type & kSilentStates) and mo.silent[i]:
      # silent states
      if sum != 0.0:
        Log.error("state %d is silent but has a non-zero emission probability", i)
    else:
      # normal states
      if abs(sum-1.0) >= GHMM_EPS_PREC:
        Log.error("sum s[%d].b[*] = %f != 1.0", i, sum)


def ghmm_dmodel_check_compatibility(mo, model_number):
  for i in range( 1,  model_number):
    if -1 == ghmm_dmodel_check_compatibel_models (mo[0], mo[i]):
      return -1

  return 0


def ghmm_dmodel_check_compatibel_models(mo, m2):
  if mo.N != m2.N:
    Log.error("different number of states (%d != %d)\n",               mo.N, m2.N)

  if mo.M != m2.M:
    Log.error("different number of possible outputs (%d != %d)\n",                  mo.M, m2.M)

  for i in range(mo.N):
    if mo.s[i].out_states != m2.s[i].out_states:
      Log.error("different number of outstates (%d != %d) in state %d.\n",                  mo.s[i].out_states, m2.s[i].out_states, i)

    for j in range(mo.s[i].out_states):
      if mo.s[i].out_id[j] != m2.s[i].out_id[j]:
        Log.error("different out_ids (%d != %d) in entry %d of state %d.\n",                      mo.s[i].out_id[j], m2.s[i].out_id[j], j, i)

  return 0


def ghmm_dmodel_generate_from_sequence(seq, seq_len, anz_symb):
  mo=ARRAY_CALLOC( 1)
  mo.N = seq_len
  mo.M = anz_symb
  # All models generated from sequences have to be LeftRight-models
  mo.model_type = kLeftRight

  # Allocate memory for all vectors
  mo.s=ARRAY_CALLOC( mo.N)
  for i in range( 0,  mo.N):
    if i == 0:
      if model_state_alloc(, mo.M, 0, 1):
        GHMM_LOG_QUEUED(LCONVERTED)
        goto STOP


    elif i == mo.N - 1:  # End state
      if model_state_alloc(mo.s + i, mo.M, 1, 0):
        GHMM_LOG_QUEUED(LCONVERTED)
        goto STOP
    else:                      # others
      if model_state_alloc(mo.s + i, mo.M, 1, 1):
        GHMM_LOG_QUEUED(LCONVERTED)
        goto STOP


  # Allocate states with the right values, the initial state and the end
  #     state extra
  for i in range( 1,  mo.N - 1):
    s = mo.s + i
    s.pi = 0.0
    s.out_states = 1
    s.in_states = 1
    s.b[seq[i]] = 1.0         # others stay 0
    (s.out_id) = i + 1
    (s.in_id) = i - 1
    (s.out_a) = (s.in_a) = 1.0


  # Initial state
  s = mo.s
  s.pi = 1.0
  s.out_states = 1
  s.in_states = 0
  s.b[seq[0]] = 1.0
  (s.out_id) = 1
  (s.out_a) = 1.0
  # No in_id and in_a

  # End state
  s = mo.s + mo.N - 1
  s.pi = 0.0
  s.out_states = 0
  s.in_states = 1
  s.b[seq[mo.N - 1]] = 1.0   # All other b's stay zero
  *(s.in_id) = mo.N - 2
  *(s.in_a) = 1.0
  # No out_id and out_a

  ghmm_dmodel_check(mo)

  return mo

 def get_random_output(mo, i, position):
  sum=0.0

  p = GHMM_RNG_UNIFORM (RNG)

  for m in range( 0,  mo.M):
    # get the right index for higher order emission models
    e_index = get_emission_index(mo, i, m, position)

    # get the probability, exit, if the index is -1
    if -1 != e_index:
      sum += mo.s[i].b[e_index]
      if sum >= p:
        break

    else:
      Log.error("State has order %d, but in the history are only %d emissions.\n",mo.order[i], position)

  if mo.M == m:
    Log.error("no valid output choosen. Are the Probabilities correct? sum: %g, p: %g\n",             sum, p)

  return (m)

def ghmm_dmodel_generate_sequences(mo, seed, global_len, seq_number, Tmax):
  int n = 0

  sq = ghmm_dseq_calloc(seq_number)

  # allocating additional fields for the state sequence in the ghmm_dseq class
  sq.states=ARRAY_CALLOC( seq_number)
  sq.states_len=ARRAY_CALLOC( seq_number)

  # A specific length of the sequences isn't given. As a model should have
  #     an end state, the konstant MAX_SEQ_LEN is used.
  if len <= 0:
    len = (int)GHMM_MAX_SEQ_LEN

  if seed > 0:
    GHMM_RNG_SET(RNG, seed)


  # initialize the emission history
  mo.emission_history = 0

  while n < seq_number:
    sq.seq[n]=ARRAY_CALLOC( len)

    # for silent models we have to allocate for the maximal possible number
    #       of lables and states
    if mo.model_type & kSilentStates:
      sq.states[n]=ARRAY_CALLOC( len * mo.N)
    else:
      sq.states[n]=ARRAY_CALLOC( len)


    pos = label_pos = 0

    # Get a random initial state i
    p = GHMM_RNG_UNIFORM(RNG)
    sum = 0.0
    for state in range( mo.N):
      sum += mo.s[state].pi
      if sum >= p:
        break


    while pos < len:
      # save the state path and label
      sq.states[n][label_pos] = state
      label_pos+=1

      # Get a random output m if the state is not a silent state
      if not (mo.model_type & kSilentStates) or not (mo.silent[state]):
 :
        m = get_random_output(mo, state, pos)
        update_emission_history(mo, m)
        sq.seq[n][pos] = m
        pos+=1


      # get next state
      p = GHMM_RNG_UNIFORM(RNG)
      if pos < mo.maxorder:
 :
        max_sum = 0.0
        for j in range( 0,  mo.s[state].out_states):
          j_id = mo.s[state].out_id[j]
          if not (mo.model_type & kHigherOrderEmissions) or mo.order[j_id] <= pos:
            max_sum += mo.s[state].out_a[j]

        if j and abs(max_sum) < GHMM_EPS_PREC:
 :
          Log.error("No possible transition from state %d "
                          "since all successor states require more history "
                          "than seen up to position: %d.",
                          state, pos)
          break

        if j:
          p *= max_sum


      sum = 0.0
      for j in range( 0,  mo.s[state].out_states):
        j_id = mo.s[state].out_id[j]
        if not (mo.model_type & kHigherOrderEmissions) or mo.order[j_id] <= pos:
 :
          sum += mo.s[state].out_a[j]
          if sum >= p:
            break



      if sum == 0.0:
 :
        Log.note("final state (%d) reached at position %d "
                        "of sequence %d", state, pos, n)
        break


      state = j_id
                               # while pos < len:
    # realocate state path and label sequence to actual size
    if mo.model_type & kSilentStates:
 :
      sq.states[n]=ARRAY_REALLOC(sq.states[n], label_pos)


    sq.seq_len[n] = pos
    sq.states_len[n] = label_pos
    n+=1
                               # while  n < seq_number :
  return (sq)
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  ghmm_dseq_free(&sq)
  return None
#undef CUR_PROC


#============================================================================

def ghmm_dmodel_likelihood(mo, sq)
:
# define CUR_PROC "ghmm_dmodel_likelihood"
  double log_p_i, log_p
  int found, i

  # printf("***  model_likelihood:\n")

  found = 0
  log_p = 0.0
  for i in range( 0,  sq.seq_number):

#         printf("sequence:\n")
#         for j in range( sq.seq_len[i]):
#                 printf("%d, ",sq.seq[i][j])
#
#         printf("\n")


    def ghmm_dmodel_logp(mo, sq.seq[i], sq.seq_len[i], &log_p_i) == -1:
 :
      GHMM_LOG_QUEUED(LCONVERTED)
      goto STOP


#         printf("\nlog_p_i = %f\n", log_p_i)

    if log_p_i != +1:
 :
      log_p += log_p_i
      found = 1

    else:
      GHMM_LOG_PRINTF(LWARN, LOC, "sequence[%d] can't be build.", i)


  if not found:
    log_p = +1.0
  return (log_p)
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  return -1
# undef CUR_PROC
                               # ghmm_dmodel_likelihood

def ghmm_dmodel_get_transition(mo, i, j)
:
# define CUR_PROC "ghmm_dmodel_get_transition"
  int out

  if mo.s and mo.s[i].out_a and mo.s[j].in_a:
 :
    for out in range( mo.s[i].out_states):
      if mo.s[i].out_id[out] == j:
        return mo.s[i].out_a[out]


  return 0.0
# undef CUR_PROC
   # ghmm_dmodel_get_transition

def ghmm_dmodel_check_transition(mo, i, j)
:
# define CUR_PROC "ghmm_dmodel_check_transition"
  int out

  if mo.s and mo.s[i].out_a and mo.s[j].in_a:
 :
    for out in range( mo.s[i].out_states):
      if mo.s[i].out_id[out] == j:
        return 1


  return 0
# undef CUR_PROC
   # ghmm_dmodel_check_transition

def ghmm_dmodel_set_transition(mo, i, j, prob)
:
# define CUR_PROC "ghmm_dmodel_set_transition"
  int in, out

  if mo.s and mo.s[i].out_a and mo.s[j].in_a:
 :
    for out in range( 0,  mo.s[i].out_states):
      if mo.s[i].out_id[out] == j:
 :
        mo.s[i].out_a[out] = prob
        # fprintf(stderr, CUR_PROC": State %d, %d, = %f\n", i, j,
        #                prob)
        break



    for in in range( 0,  mo.s[j].in_states):
      if mo.s[j].in_id[in] == i:
 :
        mo.s[j].in_a[in] = prob
        break



# undef CUR_PROC
   # ghmm_dmodel_set_transition




#============================================================================
# Some outputs
#============================================================================

def ghmm_dmodel_states_print(file, mo)
:
  int i, j
  fprintf (file, "Modelparameters: \n M = %d \t N = %d\n", mo.M, mo.N)
  for i in range( 0,  mo.N):
    fprintf (file,
             "\nState %d \n PI = %.3f \n out_states = %d \n in_states = %d \n",
             i, mo.s[i].pi, mo.s[i].out_states, mo.s[i].in_states)
    fprintf (file, " Output probability:\t")
    for j in range( 0,  mo.M):
      fprintf (file, "%.3f \t", mo.s[i].b[j])
    fprintf (file, "\n Transition probability \n")
    fprintf (file, "  Out states (Id, a):\t")
    for j in range( 0,  mo.s[i].out_states):
      fprintf (file, "(%d, %.3f) \t", mo.s[i].out_id[j], mo.s[i].out_a[j])
    fprintf (file, "\n")
    fprintf (file, "  In states (Id, a):\t")
    for j in range( 0,  mo.s[i].in_states):
      fprintf (file, "(%d, %.3f) \t", mo.s[i].in_id[j], mo.s[i].in_a[j])
    fprintf (file, "\n")

                               # ghmm_dmodel_states_print

#============================================================================

def ghmm_dmodel_A_print(file, mo, tab, separator, ending)
:
  int i, j, out_state
  for i in range( 0,  mo.N):
    out_state = 0
    fprintf (file, "%s", tab)
    if mo.s[i].out_states > 0 and mo.s[i].out_id[out_state] == 0:
 :
      fprintf (file, "%.2f", mo.s[i].out_a[out_state])
      out_state+=1

    def fprintf(file, "0.00")
    for j in range( 1,  mo.N):
      if mo.s[i].out_states > out_state and mo.s[i].out_id[out_state] == j:
 :
        fprintf (file, "%s %.2f", separator, mo.s[i].out_a[out_state])
        out_state+=1

      def fprintf(file, "%s 0.00", separator)

    fprintf (file, "%s\n", ending)

                               # ghmm_dmodel_A_print

#============================================================================

def ghmm_dmodel_B_print(file, mo, tab, separator, ending)
:
  int i, j, size

  for i in range( 0,  mo.N):
    fprintf (file, "%s", tab)
    fprintf (file, "%.2f", mo.s[i].b[0])
    if not (mo.model_type & kHigherOrderEmissions):
 :
      for j in range( 1,  mo.M):
        fprintf (file, "%s %.2f", separator, mo.s[i].b[j])
      fprintf (file, "%s\n", ending)

    else:
      size = ghmm_ipow (mo, mo.M, mo.order[i] + 1)
      for j in range( 1,  size):
        fprintf (file, "%s %.2f", separator, mo.s[i].b[j])
      fprintf (file, "%s\n", ending)


                               # ghmm_dmodel_B_print

#============================================================================

def ghmm_dmodel_Pi_print(file, mo, tab, separator, ending)
:
  int i
  fprintf (file, "%s%.2f", tab, mo.s[0].pi)
  for i in range( 1,  mo.N):
    fprintf (file, "%s %.2f", separator, mo.s[i].pi)
  fprintf (file, "%s\n", ending)
                               # ghmm_dmodel_Pi_print

def ghmm_dmodel_fix_print(file, mo, tab, separator, ending)
:
  int i
  fprintf (file, "%s%d", tab, mo.s[0].fix)
  for i in range( 1,  mo.N):
    fprintf (file, "%s %d", separator, mo.s[i].fix)
  fprintf (file, "%s\n", ending)
                               # ghmm_dmodel_Pi_print

#============================================================================

def ghmm_dmodel_A_print_transp(file, mo, tab, separator, ending)
:
# define CUR_PROC "ghmm_dmodel_A_print_transp"
  int i, j
  int *out_state

  out_state=ARRAY_CALLOC( mo.N)
  for i in range( 0,  mo.N):
    out_state[i] = 0

  for j in range( 0,  mo.N):
    fprintf (file, "%s", tab)
    if mo.s[0].out_states != 0 and mo.s[0].out_id[out_state[0]] == j:
 :
      fprintf (file, "%.2f", mo.s[0].out_a[out_state[0]])
      (out_state[0])+=1

    def fprintf(file, "0.00")
    for i in range( 1,  mo.N):
      if mo.s[i].out_states != 0 and mo.s[i].out_id[out_state[i]] == j:
 :
        fprintf (file, "%s %.2f", separator, mo.s[i].out_a[out_state[i]])
        (out_state[i])+=1

      def fprintf(file, "%s 0.00", separator)

    fprintf (file, "%s\n", ending)

STOP:     # Label STOP from ARRAY_[CM]ALLOC
  m_free (out_state)
  return
# undef CUR_PROC
                               # ghmm_dmodel_A_print_transp

#============================================================================

def ghmm_dmodel_B_print_transp(file, mo, tab, separator, ending)
:
  int i, j
  for j in range( 0,  mo.M):
    fprintf (file, "%s", tab)
    fprintf (file, "%.2f", mo.s[0].b[j])
    for i in range( 1,  mo.N):
      fprintf (file, "%s %.2f", separator, mo.s[i].b[j])
    fprintf (file, "%s\n", ending)

                               # ghmm_dmodel_B_print_transp

#============================================================================

def ghmm_dmodel_Pi_print_transp(file, mo, tab, ending)
:
  int i
  for i in range( 0,  mo.N):
    fprintf (file, "%s%.2f%s\n", tab, mo.s[i].pi, ending)
                               # ghmm_dmodel_Pi_print_transp

#============================================================================

def ghmm_dl_print(file, mo, tab, separator, ending)
:
  int i
  fprintf (file, "%s%d", tab, mo.label[0])
  for i in range( 1,  mo.N):
    fprintf (file, "%s %d", separator, mo.label[i])
  fprintf (file, "%s\n", ending)
                               # ghmm_dl_print

#============================================================================
def ghmm_dmodel_print(file, mo)
:
  fprintf (file, "HMM = :\n\tM = %d\n\tN = %d\n", mo.M, mo.N)
  fprintf (file, "\tprior = %.3f\n", mo.prior)
  fprintf (file, "\tModelType = %d\n", mo.model_type)
  fprintf (file, "\tA = matrix :\n")
  ghmm_dmodel_A_print (file, mo, "\t", ",", "")
  fprintf (file, "\t\n\tB = matrix :\n")
  ghmm_dmodel_B_print (file, mo, "\t", ",", "")
  fprintf (file, "\t\n\tPi = vector :\n")
  ghmm_dmodel_Pi_print (file, mo, "\t", ",", "")
  fprintf (file, "\t\n\tfix_state = vector :\n")
  ghmm_dmodel_fix_print (file, mo, "\t", ",", "")
  if mo.model_type & kLabeledStates:
 :
    fprintf (file, "\t\n\tlabel_state = vector :\n")
    ghmm_dl_print (file, mo, "\t", ",", "")

  fprintf (file, "\t\n\n\n")
                               # ghmm_dmodel_print

#============================================================================

#ifdef GHMM_OBSOLETE
def ghmm_dmodel_direct_print(file, mo_d, multip)
:
  int i, j
  for i in range( 0,  multip):
    fprintf (file, "HMM = :\n\tM = %d\n\tN = %d\n", mo_d.M, mo_d.N)
    fprintf (file, "\tprior = %.3f\n", mo_d.prior)
    fprintf (file, "\tA = matrix :\n")
    ighmm_cmatrix_print (file, mo_d.A, mo_d.N, mo_d.N, "\t", ",", "")
    fprintf (file, "\t\n\tB = matrix :\n")
    ighmm_cmatrix_print (file, mo_d.B, mo_d.N, mo_d.M, "\t", ",", "")
    fprintf (file, "\t\n\tPi = vector :\n")
    fprintf (file, "\t%.4f", mo_d.Pi[0])
    for j in range( 1,  mo_d.N):
      fprintf (file, ", %.4f", mo_d.Pi[j])
    fprintf (file, "\n\t\n")
    fprintf (file, "\tfix_state = vector :\n")
    fprintf (file, "\t%d", mo_d.fix_state[0])
    for j in range( 1,  mo_d.N):
      fprintf (file, ", %d", mo_d.fix_state[j])
    fprintf (file, "\n\t\n")
    fprintf (file, "\n\n")

                               # ghmm_dmodel_direct_print

#============================================================================

def ghmm_dmodel_direct_clean(mo_d, check)
:
#define CUR_PROC "ghmm_dmodel_direct_clean"
  int i
  if not mo_d:
    return
  mo_d.M = mo_d.N = 0
  mo_d.prior = -1
  if mo_d.A:
 :
    for i in range( 0,  check.r_a):
      m_free (mo_d.A[i])
    m_free (mo_d.A)

  if mo_d.B:
 :
    for i in range( 0,  check.r_b):
      m_free (mo_d.B[i])
    m_free (mo_d.B)

  if mo_d.Pi:
    m_free (mo_d.Pi)

  if mo_d.fix_state:
    m_free (mo_d.fix_state)


  mo_d.A = mo_d.B = None
  mo_d.Pi = None
  mo_d.fix_state = None
#undef CUR_PROC
                               # ghmm_dmodel_direct_clean

#============================================================================

def ghmm_dmodel_direct_check_data(mo_d, check)
:
#define CUR_PROC "ghmm_dmodel_direct_check_data"
  char *str
  if check.r_a != mo_d.N or check.c_a != mo_d.N:
 :
    str = ighmm_mprintf (None, 0, "Incompatible dim. A (%d X %d) and N (%d)\n",
                   check.r_a, check.c_a, mo_d.N)
    GHMM_LOG(LCONVERTED, str)
    m_free (str)
    return (-1)

  if check.r_b != mo_d.N or check.c_b != mo_d.M:
 :
    str = ighmm_mprintf (None, 0,
            "Incompatible dim. B (%d X %d) and N X M (%d X %d)\n",
            check.r_b, check.c_b, mo_d.N, mo_d.M)
    GHMM_LOG(LCONVERTED, str)
    m_free (str)
    return (-1)

  if check.len_pi != mo_d.N:
 :
    str = ighmm_mprintf (None, 0, "Incompatible dim. Pi (%d) and N (%d)\n",
                   check.len_pi, mo_d.N)
    GHMM_LOG(LCONVERTED, str)
    m_free (str)
    return (-1)

  if check.len_fix != mo_d.N:
 :
    str = ighmm_mprintf (None, 0, "Incompatible dim. fix_state (%d) and N (%d)\n",
                   check.len_fix, mo_d.N)
    GHMM_LOG(LCONVERTED, str)
    m_free (str)
    return (-1)


  return 0
#undef CUR_PROC
                               # ghmm_dmodel_direct_check_data
#endif # GHMM_OBSOLETE


#============================================================================
# XXX symmetric not implemented yet
def ghmm_dmodel_prob_distance(m0, m, maxT, symmetric, verbose)
:
#define CUR_PROC "ghmm_dmodel_prob_distance"

#define STEPS 40

  double p0, p
  double d = 0.0
  double *d1
  ghmm_dseq *seq0 = None
  ghmm_dseq *tmp = None
  ghmm_dmodel *mo1, *mo2
  int i, t, a, k
  int true_len
  int true_number
  int left_to_right = 0
  long total, index
  int step_width = 0
  int steps = 1

  # printf("***  model_prob_distance:\n")

  if verbose:
 :                # If we are doing it verbosely we want to have 40 steps
    step_width = maxT / 40
    steps = STEPS

  else                          # else just one
    step_width = maxT

  d1=ARRAY_CALLOC( steps)

  mo1 = m0
  mo2 = m

  for (k = 0 k < 2 k++) :     # Two passes for the symmetric case

    # seed = 0 . no reseeding. Call  ghmm_rng_timeseed(RNG) externally
    seq0 = ghmm_dmodel_generate_sequences (mo1, 0, maxT + 1, 1, maxT + 1)



    if seq0 == None:
 :
      GHMM_LOG(LCONVERTED, " generate_sequences failed not ")
      goto STOP


    if seq0.seq_len[0] < maxT:
 :      # There is an absorbing state

      # NOTA BENE: Assumpting the model delivers an explicit end state,
      #         the condition of a fix initial state is removed.

      # For now check that Pi puts all weight on state
      #
      #         t = 0
      #         for i in range( 0,  mo1.N):
      #         if mo1.s[i].pi > 0.001:
      #         t+=1
      #
      #         if t > 1:
 :
      #         GHMM_LOG(LCONVERTED, "ERROR: No proper left-to-right model. Multiple start states")
      #         goto STOP
      #

      left_to_right = 1
      total = seq0.seq_len[0]

      while total <= maxT:

        # create a additional sequences at once
        a = (maxT - total) / (total / seq0.seq_number) + 1
        # printf("total=%d generating %d", total, a)
        tmp = ghmm_dmodel_generate_sequences (mo1, 0, 0, a, a)
        if tmp == None:
 :
          GHMM_LOG(LCONVERTED, " generate_sequences failed not ")
          goto STOP

        ghmm_dseq_free (&tmp)
        ghmm_dseq_add (seq0, tmp)

        total = 0
        for i in range( 0,  seq0.seq_number):
          total += seq0.seq_len[i]



    if left_to_right:
 :

      for t in range( step_width, i = 0,  maxT):

        index = 0
        total = seq0.seq_len[0]

        # Determine how many sequences we need to get a total of t
        #           and adjust length of last sequence to obtain total of
        #           exactly t

        while total < t:
          index+=1
          total += seq0.seq_len[index]


        true_len = seq0.seq_len[index]
        true_number = seq0.seq_number

        if (total - t) > 0:
          seq0.seq_len[index] = total - t
        seq0.seq_number = index

        p0 = ghmm_dmodel_likelihood (mo1, seq0)
        if p0 == +1 or p0 == -1:
 :     # errornot
          GHMM_LOG(LCONVERTED, "problem: ghmm_dmodel_likelihood failed not ")
          goto STOP

        p = ghmm_dmodel_likelihood (mo2, seq0)
        if p == +1 or p == -1:
 :       # what shall we do now?
          GHMM_LOG(LCONVERTED, "problem: ghmm_dmodel_likelihood failed not ")
          goto STOP


        d = 1.0 / t * (p0 - p)

        if symmetric:
 :
          if k == 0:
            # save d
            d1[i] = d
          else:
            # calculate d
            d = 0.5 * (d1[i] + d)



        if verbose and (not symmetric or k == 1):
          printf ("%d\t%f\t%f\t%f\n", t, p0, p, d)

        seq0.seq_len[index] = true_len
        seq0.seq_number = true_number



    else:

      true_len = seq0.seq_len[0]

      for t in range( step_width, i = 0,  maxT):
        seq0.seq_len[0] = t

        p0 = ghmm_dmodel_likelihood (mo1, seq0)
        # printf("   P(O|m1) = %f\n",p0)
        if p0 == +1:
 :         # errornot
          GHMM_LOG(LCONVERTED, "seq0 can't be build from mo1not ")
          goto STOP

        p = ghmm_dmodel_likelihood (mo2, seq0)
        # printf("   P(O|m2) = %f\n",p)
        if p == +1:
 :          # what shall we do now?
          GHMM_LOG(LCONVERTED, "problem: seq0 can't be build from mo2not ")
          goto STOP


        d = (1.0 / t) * (p0 - p)

        if symmetric:
 :
          if k == 0:
            # save d
            d1[i] = d
          else:
            # calculate d
            d = 0.5 * (d1[i] + d)



        if verbose and (not symmetric or k == 1):
          printf ("%d\t%f\t%f\t%f\n", t, p0, p, d)


      seq0.seq_len[0] = true_len


    if symmetric:
 :
      ghmm_dseq_free (&seq0)
      mo1 = m
      mo2 = m0

    else
      break

                               # k = 1,2

  ghmm_dseq_free (&seq0)
  m_free (d1)
  return d

STOP:     # Label STOP from ARRAY_[CM]ALLOC
  ghmm_dseq_free (&seq0)
  m_free (d1)
  return (0.0)
#undef CUR_PROC



#============================================================================

def ghmm_dstate_clean(my_state)
:
#define CUR_PROC "ghmm_dstate_clean"

  mes_check_ptr (my_state, return)

  if my_state.b:
    m_free (my_state.b)

  if my_state.out_id:
    m_free (my_state.out_id)

  if my_state.in_id:
    m_free (my_state.in_id)

  if my_state.out_a:
    m_free (my_state.out_a)

  if my_state.in_a:
    m_free (my_state.in_a)

  my_state.pi = 0
  my_state.b = None
  my_state.out_id = None
  my_state.in_id = None
  my_state.out_a = None
  my_state.in_a = None
  my_state.out_states = 0
  my_state.in_states = 0
  my_state.fix = 0

#undef CUR_PROC
                               # ghmm_dstate_clean



#==========================  Labeled HMMs  ================================

def ghmm_dmodel_label_generate_sequences(mo, seed, global_len, seq_number, Tmax)
:
#define CUR_PROC "ghmm_dmodel_label_generate_sequences"

  ghmm_dseq *sq = None
  int state
  int j, m, j_id
  double p, sum, max_sum
  int len = global_len
  int n = 0
  int pos, label_pos

  sq = ghmm_dseq_calloc(seq_number)
  if not sq:
 :
    GHMM_LOG_QUEUED(LCONVERTED)
    goto STOP


  # allocating additional fields for the state sequence in the ghmm_dseq class
  sq.states=ARRAY_CALLOC( seq_number)
  sq.states_len=ARRAY_CALLOC( seq_number)

  # allocating additional fields for the labels in the ghmm_dseq class
  sq.state_labels=ARRAY_CALLOC( seq_number)
  sq.state_labels_len=ARRAY_CALLOC( seq_number)

  # A specific length of the sequences isn't given. As a model should have
  #     an end state, the konstant MAX_SEQ_LEN is used.
  if len <= 0:
    len = (int)GHMM_MAX_SEQ_LEN

  if seed > 0:
 :
    GHMM_RNG_SET(RNG, seed)


  # initialize the emission history
  mo.emission_history = 0

  while n < seq_number:
    sq.seq[n]=ARRAY_CALLOC( len)

    # for silent models we have to allocate for the maximal possible number
    #       of lables and states
    if mo.model_type & kSilentStates:
 :
      sq.states[n]=ARRAY_CALLOC( len * mo.N)
      sq.state_labels[n]=ARRAY_CALLOC( len * mo.N)

     else:
      sq.states[n]=ARRAY_CALLOC( len)
      sq.state_labels[n]=ARRAY_CALLOC( len)


    pos = label_pos = 0

    # Get a random initial state i
    p = GHMM_RNG_UNIFORM(RNG)
    sum = 0.0
    for state in range( mo.N):
      sum += mo.s[state].pi
      if sum >= p:
        break


    while pos < len:
      # save the state path and label
      sq.states[n][label_pos] = state
      sq.state_labels[n][label_pos] = mo.label[state]
      label_pos+=1

      # Get a random output m if the state is not a silent state
      if not (mo.model_type & kSilentStates) or not (mo.silent[state]):
 :
        m = get_random_output(mo, state, pos)
        update_emission_history(mo, m)
        sq.seq[n][pos] = m
        pos+=1


      # get next state
      p = GHMM_RNG_UNIFORM(RNG)
      if pos < mo.maxorder:
 :
        max_sum = 0.0
        for j in range( 0,  mo.s[state].out_states):
          j_id = mo.s[state].out_id[j]
          if not (mo.model_type & kHigherOrderEmissions) or mo.order[j_id] < pos:
            max_sum += mo.s[state].out_a[j]

        if j and abs(max_sum) < GHMM_EPS_PREC:
 :
          Log.error("No possible transition from state %d "
                          "since all successor states require more history "
                          "than seen up to position: %d.",
                          state, pos)
          break

        if j:
          p *= max_sum


      sum = 0.0
      for j in range( 0,  mo.s[state].out_states):
        j_id = mo.s[state].out_id[j]
        if not (mo.model_type & kHigherOrderEmissions) or mo.order[j_id] < pos:
 :
          sum += mo.s[state].out_a[j]
          if sum >= p:
            break



      if sum == 0.0:
 :
        Log.note("final state (%d) reached at position %d "
                        "of sequence %d", state, pos, n)
        break


      state = j_id
                               # while pos < len:
    # realocate state path and label sequence to actual size
    if mo.model_type & kSilentStates:
 :
      sq.states[n]=ARRAY_REALLOC(sq.states[n], label_pos)
      sq.state_labels[n]=ARRAY_REALLOC(sq.state_labels[n], label_pos)


    sq.seq_len[n] = pos
    sq.states_len[n] = label_pos
    sq.state_labels_len[n] = label_pos
    n+=1
                               # while  n < seq_number :
  return (sq)
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  ghmm_dseq_free(&sq)
  return None
#undef CUR_PROC



#-
# Scales the output and transitions probs of all states in a given model
def ghmm_dmodel_normalize(mo)
:
#define CUR_PROC "ghmm_dmodel_normalize"
  double pi_sum=0.0
  int i, j, m, j_id, i_id=0, res=0
  int size = 1

  for i in range( 0,  mo.N):
    if mo.s[i].pi >= 0.0:
      pi_sum += mo.s[i].pi
    else
      mo.s[i].pi = 0.0

    # check model_type before using state order
    if mo.model_type & kHigherOrderEmissions:
      size = ghmm_ipow (mo, mo.M, mo.order[i])

    # normalize transition probabilities
    def ighmm_cvector_normalize(mo.s[i].out_a, mo.s[i].out_states) == -1:
 :
      res = -1

    # for every outgoing probability update the corrosponding incoming probability
    for j in range( 0,  mo.s[i].out_states):
      j_id = mo.s[i].out_id[j]
      for m in range( 0,  mo.s[j_id].in_states):
        if i == mo.s[j_id].in_id[m]:
 :
          i_id = m
          break


      if i_id == mo.s[j_id].in_states:
 :
        Log.error("Outgoing transition from state %d to \
           state %d has no corresponding incoming transition.", i, j_id)
        return -1

      mo.s[j_id].in_a[i_id] = mo.s[i].out_a[j]

    # normalize emission probabilities, but not for silent states
    if not ((mo.model_type & kSilentStates) and mo.silent[i]):
 :
      for m in range( 0,  size):
        def ighmm_cvector_normalize(&(mo.s[i].b[m * mo.M]), mo.M) == -1:
          res = -1



  for i in range( 0,  mo.N):
    mo.s[i].pi /= pi_sum

  return res
#undef CUR_PROC
                               # ghmm_dmodel_normalize


#-
def ghmm_dmodel_add_noise(mo, level, seed)
:
#define CUR_PROC "model_add_noise_A"

  int h, i, j, hist
  int size = 1

  if level > 1.0:
    level = 1.0

  for i in range( 0,  mo.N):
    for j in range( 0,  mo.s[i].out_states):
      # add noise only to out_a, in_a is updated on normalisation
      mo.s[i].out_a[j] *= (1 - level) + (GHMM_RNG_UNIFORM (RNG) * 2 * level)

    if mo.model_type & kHigherOrderEmissions:
      size = ghmm_ipow (mo, mo.M, mo.order[i])
    for hist in range( 0,  size):
      for h in range( hist * mo.M,  hist * mo.M + mo.M):
        mo.s[i].b[h] *= (1 - level) + (GHMM_RNG_UNIFORM (RNG) * 2 * level)


  def ghmm_dmodel_normalize(mo)

#undef CUR_PROC


#-
 def ghmm_dstate_transition_add(s, start, dest, prob)
:
#define CUR_PROC "ghmm_dmodel_transition_add"

  int i

  # resize the arrays
  s[dest].in_id=ARRAY_REALLOC(s[dest].in_id, s[dest].in_states + 1)
  s[dest].in_a=ARRAY_REALLOC(s[dest].in_a, s[dest].in_states + 1)
  s[start].out_id=ARRAY_REALLOC(s[start].out_id, s[start].out_states + 1)
  s[start].out_a=ARRAY_REALLOC(s[start].out_a, s[start].out_states + 1)

  s[dest].in_states += 1
  s[start].out_states += 1

  # search the right place to insert while moving greater entrys one field back
  for (i = s[start].out_states - 1 i >= 0 i--) :
    if i == 0 or dest > s[start].out_id[i - 1]:
 :
      s[start].out_id[i] = dest
      s[start].out_a[i] = prob
      break

    else:
      s[start].out_id[i] = s[start].out_id[i - 1]
      s[start].out_a[i] = s[start].out_a[i - 1]



  # search the right place to insert while moving greater entrys one field back
  for (i = s[dest].in_states - 1 i >= 0 i--)
    if i == 0 or start > s[dest].in_id[i - 1]:
 :
      s[dest].in_id[i] = start
      s[dest].in_a[i] = prob
      break

    else:
      s[dest].in_id[i] = s[dest].in_id[i - 1]
      s[dest].in_a[i] = s[dest].in_a[i - 1]


  return 0
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  return -1
#undef CUR_PROC


#-
 def ghmm_dstate_transition_del(s, start, dest)
:
#define CUR_PROC "ghmm_dmodel_transition_del"

  int i, j

  # search ...
  for (j = 0 dest != s[start].out_id[j] j++)
    if j == s[start].out_states:
 :
      GHMM_LOG(LCONVERTED, "No such transition")
      return -1

  # ... and replace outgoing
  for i in range( j + 1,  s[start].out_states):
    s[start].out_id[i - 1] = s[start].out_id[i]
    s[start].out_a[i - 1] = s[start].out_a[i]


  # search ...
  for (j = 0 start != s[dest].in_id[j] j++)
    if j == s[dest].in_states:
 :
      GHMM_LOG(LCONVERTED, "No such transition")
      return -1

  # ... and replace incoming
  for i in range( j + 1,  s[dest].in_states):
    s[dest].in_id[i - 1] = s[dest].in_id[i]
    s[dest].in_a[i - 1] = s[dest].in_a[i]


  # reset number
  s[dest].in_states -= 1
  s[start].out_states -= 1

  # free memory
  s[dest].in_id=ARRAY_REALLOC(s[dest].in_id, s[dest].in_states)
  s[dest].in_a=ARRAY_REALLOC(s[dest].in_a, s[dest].in_states)
  s[start].out_id=ARRAY_REALLOC(s[start].out_id, s[start].out_states)
  s[start].out_a=ARRAY_REALLOC(s[start].out_a, s[start].out_states)

  return 0
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  return -1
#undef CUR_PROC


#-
#*
#   Allocates a new ghmm_dbackground class and assigs the arguments to
#   the respective fields. Note: The arguments need allocation outside of this
#   function.
#
#   @return     :               0 on success, -1 on error
#   @param mo   :               one model
#   @param cur  :               a id of a state
#   @param times:               number of times the state cur is at least evaluated
#
def ghmm_dmodel_duration_apply(mo, cur, times)
:
#define CUR_PROC "ghmm_dmodel_duration_apply"

  int i, j, last, size, failed=0

  if mo.model_type & kSilentStates:
 :
    GHMM_LOG(LCONVERTED, "Sorry, apply_duration doesn't support silent states yet\n")
    return -1


  last = mo.N
  mo.N += times - 1

  mo.s=ARRAY_REALLOC(mo.s, mo.N)
  if mo.model_type & kSilentStates:
 :
    mo.silent=ARRAY_REALLOC(mo.silent, mo.N)
    mo.topo_order=ARRAY_REALLOC(mo.topo_order, mo.N)

  if mo.model_type & kTiedEmissions:
    mo.tied_to=ARRAY_REALLOC(mo.tied_to, mo.N)
  if mo.model_type & kBackgroundDistributions:
    mo.background_id=ARRAY_REALLOC(mo.background_id, mo.N)

  size = ghmm_ipow (mo, mo.M, mo.order[cur] + 1)
  for i in range( last,  mo.N):
    # set the new state
    mo.s[i].pi = 0.0
    mo.order[i] = mo.order[cur]
    mo.s[i].fix = mo.s[cur].fix
    mo.label[i] = mo.label[cur]
    mo.s[i].in_a = None
    mo.s[i].in_id = None
    mo.s[i].in_states = 0
    mo.s[i].out_a = None
    mo.s[i].out_id = None
    mo.s[i].out_states = 0

    mo.s[i].b=ARRAY_MALLOC( size)
    for j in range( 0,  size):
      mo.s[i].b[j] = mo.s[cur].b[j]

    if mo.model_type & kSilentStates:
      mo.silent[i] = mo.silent[cur]
      # XXX what to do with topo_order
      #         mo.topo_order[i] = ????????????

    if mo.model_type & kTiedEmissions:
      # XXX is there a clean solution for tied states?
      #         what if the current state is a tie group leader?
      #         the last added state should probably become
      #         the new tie group leader
      mo.tied_to[i] = kUntied
    if mo.model_type & kBackgroundDistributions:
      mo.background_id[i] = mo.background_id[cur]


  # move the outgoing transitions to the last state
  while mo.s[cur].out_states > 0:
    if mo.s[cur].out_id[0] == cur:
 :
      ghmm_dstate_transition_add (mo.s, mo.N - 1, mo.N - 1, mo.s[cur].out_a[0])
      ghmm_dstate_transition_del (mo.s, cur, mo.s[cur].out_id[0])

    else:
      ghmm_dstate_transition_add (mo.s, mo.N - 1, mo.s[cur].out_id[0],
                            mo.s[cur].out_a[0])
      ghmm_dstate_transition_del (mo.s, cur, mo.s[cur].out_id[0])



  # set the linear transitions through all added states
  ghmm_dstate_transition_add (mo.s, cur, last, 1.0)
  for i in range( last + 1,  mo.N):
    ghmm_dstate_transition_add (mo.s, i - 1, i, 1.0)


  def ghmm_dmodel_normalize(mo):
    goto STOP

  return 0
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  # Fail hard if these realloc fail. They shouldn't because we have the memory
  #     and try only to clean upnot
  if failed++:
    exit (1)

  mo.s=ARRAY_REALLOC(mo.s, last)
  mo.tied_to=ARRAY_REALLOC(mo.tied_to, last)
  mo.background_id=ARRAY_REALLOC(mo.background_id, last)

  mo.N = last
  return -1
#undef CUR_PROC


#-
def ghmm_dbackground_alloc(n, m, orders, B) :
#define CUR_PROC "ghmm_dbackground_alloc"
  ghmm_dbackground *ptbackground

  ptbackground=ARRAY_CALLOC( 1)

  # initialize name
  ptbackground.name=ARRAY_CALLOC( n)
  int i
  for i in range( 0,  n):
    ptbackground.name[i] = None


  ptbackground.n = n
  ptbackground.m = m
  if orders:
    ptbackground.order = orders
  if B:
    ptbackground.b = B

  return ptbackground
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  return None
#undef CUR_PROC


#-
def ghmm_dbackground_copy(bg)
:
#define CUR_PROC "ghmm_dbackground_copy"
  int i, j, b_i_len
  int *new_order
  double **new_b

  new_order=ARRAY_MALLOC( bg.n)
  new_b=ARRAY_CALLOC( bg.n)

  for i in range( 0,  bg.n):
    new_order[i] = bg.order[i]
    b_i_len = pow (bg.m, bg.order[i] + 1)
    new_b[i]=ARRAY_CALLOC( b_i_len)
    for j in range( 0,  b_i_len):
      new_b[i][j] = bg.b[i][j]



  ghmm_dbackground *tmp = ghmm_dbackground_alloc (bg.n, bg.m, new_order,
                                               new_b)

  for i in range( 0,  bg.n):
    if bg.name[i]) strcpy(tmp.name[i], bg.name[i]:


  return tmp

STOP:     # Label STOP from ARRAY_[CM]ALLOC

  return None

#undef CUR_PROC



#-
def ghmm_dmodel_background_apply(mo, background_weight)
:
# define CUR_PROC "ghmm_dmodel_apply_background"

  int i, j, size

  if not (mo.model_type & kBackgroundDistributions):
 :
    GHMM_LOG(LCONVERTED, "Error: No background distributions")
    return -1


  for i in range(mo.N):
    if mo.background_id[i] != kNoBackgroundDistribution:
 :
      if mo.model_type & kHigherOrderEmissions:
 :
        if mo.order[i] != mo.bp.order[mo.background_id[i]]:
 :
          Log.error("State (%d) and background order (%d) "
                               "do not match in state %d. Background_id = %d",
                               mo.order[i],
                               mo.bp.order[mo.background_id[i]], i,
                               mo.background_id[i])
          return -1

        # XXX Cache in ghmm_dbackground
        size = pow( mo.M, mo.order[i]+1)
        for j in range(size):
          mo.s[i].b[j] = (1.0 - background_weight[i]) * mo.s[i].b[j]
            + background_weight[i] * mo.bp.b[mo.background_id[i]][j]
       else:
        if mo.bp.order[mo.background_id[i]] != 0:
 :
          GHMM_LOG(LCONVERTED, "Error: State and background order do not match\n")
          return -1

        for j in range(mo.M):
          mo.s[i].b[j] = (1.0 - background_weight[i]) * mo.s[i].b[j]
            + background_weight[i] * mo.bp.b[mo.background_id[i]][j]




  return 0
#undef CUR_PROC
                               # ghmm_dmodel_apply_background


#-
def ghmm_dmodel_get_uniform_background(mo, sq)
:
# define CUR_PROC "get_background"

  int h, i, j, m, t, n=0
  int e_index, size
  double sum=0.0

  if not (mo.model_type & kBackgroundDistributions):
 :
    GHMM_LOG(LCONVERTED, "Error: Model has no background distribution")
    return -1


  mo.bp = None
  mo.background_id=ARRAY_MALLOC( mo.N)

  # create a background distribution for each state
  for i in range( 0,  mo.N):
    mo.background_id[i] = mo.order[i]


  # allocate
  mo.bp=ARRAY_CALLOC( 1)
  mo.bp.order=ARRAY_CALLOC( mo.maxorder)

  # set number of distributions
  mo.bp.n = mo.maxorder

  # set br.order
  for i in range( 0,  mo.N):
    if mo.background_id[i] != kNoBackgroundDistribution:
      mo.bp.order[mo.background_id[i]] = mo.order[i]

  # allocate and initialize br.b with zeros
  mo.bp.b=ARRAY_CALLOC( mo.bp.n)

  for i in range( 0,  mo.bp.n):
    mo.bp.b[i]=ARRAY_MALLOC( ghmm_ipow (mo, mo.M, mo.bp.order[i] + 1))

  for i in range( 0,  mo.bp.n):

    # find a state with the current order
    for j in range( 0,  mo.N):
      if mo.bp.order[i] == mo.order[j]:
        break

    # initialize with ones as psoudocounts
    size = ghmm_ipow (mo, mo.M, mo.bp.order[n] + 1)
    for m in range( 0,  size):
      mo.bp.b[i][m] = 1.0

    for n in range( 0,  sq.seq_number):

      for t in range( 0,  mo.bp.order[i]):
        update_emission_history (mo, sq.seq[n][t])

      for t in range( mo.bp.order[i],  sq.seq_len[n]):

        e_index = get_emission_index (mo, j, sq.seq[n][t], t)
        if -1 != e_index:
          mo.bp.b[i][e_index]+=1



    # normalise
    size = ghmm_ipow (mo, mo.M, mo.bp.order[n])
    for (h = 0 h < size h += mo.M) :
      for m in range( h,  h + mo.M):
        sum += mo.bp.b[i][m]
      for m in range( h,  h + mo.M):
        mo.bp.b[i][m] /= sum




  return 0

STOP:     # Label STOP from ARRAY_[CM]ALLOC


  return -1
# undef CUR_PROC
                               # end get_background


#============================================================================
def ghmm_dmodel_distance(mo, m2) :
#define CUR_PROC "model_distances"

  int i, j, number=0
  double tmp, distance=0.0

#   if not ghmm_dmodel_check_compatibility(mo, m2):

#     exit(1)
#   if not ghmm_dmodel_check(mo):

#     exit(1)
#   if not ghmm_dmodel_check(m2):

#     exit(1)


  # PI
  for i in range(mo.N):
    tmp = mo.s[i].pi - m2.s[i].pi
    distance += tmp*tmp
    ++number

  for i in range(mo.N):
    # A
    for j in range(mo.s[i].out_states):
      tmp = mo.s[i].out_a[j] - m2.s[i].out_a[j]
      distance += tmp*tmp
      ++number

    # B
    for j in range(pow( mo.M, mo.order[i]+1)):
      tmp = mo.s[i].b[j] - m2.s[i].b[j]
      distance += tmp*tmp
      ++number



  return (distance/number)
#undef CUR_PROC


#============================================================================
def ghmm_dmodel_xml_read(filename, mo_number) :
#define CUR_PROC "ghmm_dmodel_xml_read"
  ghmm_xmlfile* f
  ghmm_dmodel** mo

  f = ghmm_xmlfile_parse(filename)
  if not f and (f.modelType & kDiscreteHMM:
      and not (f.modelType & (kPairHMM+kTransitionClasses))) :
    Log.error("wrong model type, model in file is not a plain discrete model")
    goto STOP

  mo_number = f.noModels
  mo = f.model.d

  free(f) # XXX - by now, we free f
  return mo
STOP:
  return None
#undef CUR_PROC



#============================================================================
def ghmm_dmodel_xml_write(mo, file, mo_number) :
#define CUR_PROC "ghmm_dmodel_xml_write"

  ghmm_xmlfile* f

  f=ARRAY_MALLOC(1)
  f.noModels = mo_number
  f.modelType = mo[0].model_type
  f.model.d = mo
  ghmm_xmlfile_write(f, file)
  free(f)
  return 0
STOP:
  return -1
#undef CUR_PROC



#===================== E n d   o f  f i l e  "model.c"       ===============
