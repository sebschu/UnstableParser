#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat, Sebastian Schuster (the server component)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import re
import os
import sys
import codecs
import tempfile
from argparse import ArgumentParser

import numpy as np

from flask import Flask, jsonify, request, Response
from corenlp_protobuf import Document, parseFromDelimitedString, writeToDelimitedString

from parser.misc.colors import ctext
from parser.misc.mst import nonprojective, argmax

from parser import Configurable
from parser import Network

network =  None

# TODO make the pretrained vocab names a list given to TokenVocab
#***************************************************************
# Set up the argparser
argparser = ArgumentParser('Network')
argparser.add_argument('--save_dir', required=True)
subparsers = argparser.add_subparsers()
section_names = set()
# --section_name opt1=value1 opt2=value2 opt3=value3
with codecs.open('config/defaults.cfg') as f:
  section_regex = re.compile('\[(.*)\]')
  for line in f:
    match = section_regex.match(line)
    if match:
      section_names.add(match.group(1).lower().replace(' ', '_'))



#=================================================================
# Web service
#-----------------------------------------------------------------
app = Flask(__name__)

@app.route('/ping/')
def handle_ping():
  return "pong"

@app.route('/annotate/',  methods=['POST'])
def annotate():
  msg = request.data
  # Do the annotation
  doc = Document()
  parseFromDelimitedString(doc, msg)

  _, input_file = tempfile.mkstemp()
  f = codecs.open(input_file, 'w', encoding='utf-8', errors='ignore')
  for sentence in doc.sentence:
    f.write(_to_conllu(sentence))
  f.close()

  sents, probs, parseset = network.parse_online(input_file)
  _fill_parse_annotations(doc, probs, parseset)

  os.remove(input_file)

  with io.BytesIO() as stream:
    writeToDelimitedString(doc, stream)
    msg = stream.getvalue()

  return Response(msg, mimetype="application/x-protobuf")

def _to_conllu(sentence_ann):
  conllu_lines = []
  for idx, token in enumerate(sentence_ann.token):
    lemma = token.lemma if len(token.lemma) > 0 else  "_"
    uPOS = token.coarseTag if len(token.coarseTag) > 0 else "_"
    conllu_string = "%d\t%s\t%s\t%s\t%s\t_\t_\t_\t_\t_" % ((idx + 1),
                                                           token.word,
                                                           lemma,
                                                           uPOS,
                                                           token.pos)
    conllu_lines.append(conllu_string)

  #append extra empty line at the end
  conllu_lines.append("\n")
  return "\n".join(conllu_lines)

def _fill_parse_annotations(ann, probs, parseset):
  arc_probs = [arc_prob for batch in probs for arc_prob in batch[0]]
  rel_probs = [rel_prob for batch in probs for rel_prob in batch[1]]
  tokens_to_keep = [weight for batch in probs for weight in batch[2]]

  j = 0
  for i in parseset.multibucket.inv_idxs():
    #TODO: figure out why this happens
    if j >= len(ann.sentence):
      break
    sentence = ann.sentence[j]
    arc_prob, rel_prob, weights = arc_probs[i], rel_probs[i], tokens_to_keep[i]
    sequence_length = int(np.sum(weights))+1
    arc_prob = arc_prob[:sequence_length][:,:sequence_length]
    arc_preds = nonprojective(arc_prob)
    arc_preds_one_hot = np.zeros([rel_prob.shape[0], rel_prob.shape[2]])
    arc_preds_one_hot[np.arange(len(arc_preds)), arc_preds] = 1.
    rel_preds = np.argmax(np.einsum('nrb,nb->nr', rel_prob, arc_preds_one_hot), axis=1)
    for token, arc_pred, rel_pred, weight in zip(sentence.token, arc_preds[1:], rel_preds[1:], weights[1:]):
      token.conllUMisc = "{0}:{1}".format(parseset._nlp_model.vocabs['heads'][arc_pred], parseset._nlp_model.vocabs['rels'][rel_pred])
    j += 1


#================================================================
# Parse server/annotator
#----------------------------------------------------------------

def parse_server(save_dir, **kwargs):
  global network
  kwargs['config_file'] = os.path.join(save_dir, 'config.cfg')
  kwargs['is_evaluation'] = True
  network = Network(**kwargs)
  network.preload()

  print(ctext('Starting annotation server...', 'bright_green'))
  app.run()

  return

#----------------------------------------------------------------

parse_server_parser = subparsers.add_parser('parse_server')
parse_server_parser.set_defaults(action=parse_server)
for section_name in section_names:
  parse_server_parser.add_argument('--'+section_name, nargs='+')
parse_server_parser.add_argument('--output_file')
parse_server_parser.add_argument('--output_dir')



#***************************************************************
# Parse the arguments
kwargs = vars(argparser.parse_args())
action = kwargs.pop('action')
save_dir = kwargs.pop('save_dir')
kwargs = {key: value for key, value in kwargs.iteritems() if value is not None}
for section, values in kwargs.iteritems():
  if section in section_names:
    values = [value.split('=', 1) for value in values]
    kwargs[section] = {opt: value for opt, value in values}
if 'default' not in kwargs:
  kwargs['default'] = {}
kwargs['default']['save_dir'] = save_dir
action(save_dir, **kwargs)
