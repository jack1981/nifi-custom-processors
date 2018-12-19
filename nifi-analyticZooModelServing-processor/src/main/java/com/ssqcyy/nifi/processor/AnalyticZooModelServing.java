/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.ssqcyy.nifi.processor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.nifi.annotation.behavior.InputRequirement;
import org.apache.nifi.annotation.documentation.CapabilityDescription;
import org.apache.nifi.annotation.documentation.Tags;
import org.apache.nifi.annotation.lifecycle.OnScheduled;
import org.apache.nifi.components.PropertyDescriptor;
import org.apache.nifi.flowfile.FlowFile;
import org.apache.nifi.logging.ComponentLog;
import org.apache.nifi.processor.AbstractProcessor;
import org.apache.nifi.processor.ProcessContext;
import org.apache.nifi.processor.ProcessSession;
import org.apache.nifi.processor.ProcessorInitializationContext;
import org.apache.nifi.processor.Relationship;
import org.apache.nifi.processor.exception.ProcessException;
import org.apache.nifi.processor.util.StandardValidators;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

@InputRequirement(InputRequirement.Requirement.INPUT_REQUIRED)
@Tags({ "spark", "machinelearning", "realtime", "modelserving" })
@CapabilityDescription("Execute a real time User-item model serving for analytic zoo lib")
public class AnalyticZooModelServing extends AbstractProcessor {

	public static final PropertyDescriptor USER_ID = new PropertyDescriptor.Builder().name("user-id")
			.displayName("User Id").description("The userId for model serving.")
			.addValidator(StandardValidators.NON_EMPTY_VALIDATOR).required(true).expressionLanguageSupported(true)
			.build();

	public static final PropertyDescriptor ITEM_ID = new PropertyDescriptor.Builder().name("item-id")
			.displayName("Item Id").description("The itemId for model serving.")
			.addValidator(StandardValidators.NON_EMPTY_VALIDATOR).required(true).expressionLanguageSupported(true)
			.build();
	
	public static final PropertyDescriptor MODEL_FILE_PATH = new PropertyDescriptor.Builder().name("model-file-path")
			.displayName("Model File Path").description("The path of model file")
			.addValidator(StandardValidators.NON_EMPTY_VALIDATOR).required(true).expressionLanguageSupported(true)
			.build();

	public static final Relationship REL_SUCCESS = new Relationship.Builder().name("success")
			.description("FlowFiles that are successfully processed are sent to this relationship").build();

	public static final Relationship REL_FAILURE = new Relationship.Builder().name("failure")
			.description("FlowFiles are routed to this relationship when they cannot be parsed").build();

	private volatile List<PropertyDescriptor> properties;
	private volatile Set<Relationship> relationships;

	private NueralCFModel rcm;

	@Override
	public void init(final ProcessorInitializationContext context) {
		List<PropertyDescriptor> properties = new ArrayList<>();
		properties.add(USER_ID);
		properties.add(ITEM_ID);
		properties.add(MODEL_FILE_PATH);
		this.properties = Collections.unmodifiableList(properties);

		Set<Relationship> relationships = new HashSet<>();
		relationships.add(REL_SUCCESS);
		relationships.add(REL_FAILURE);
		this.relationships = Collections.unmodifiableSet(relationships);
	}

	@Override
	public Set<Relationship> getRelationships() {
		return relationships;
	}

	@Override
	protected List<PropertyDescriptor> getSupportedPropertyDescriptors() {
		return properties;
	}

	@Override
	protected PropertyDescriptor getSupportedDynamicPropertyDescriptor(final String propertyDescriptorName) {
		return new PropertyDescriptor.Builder()
				.description("Specifies the value for '" + propertyDescriptorName + "' for model serving job")
				.name(propertyDescriptorName).dynamic(true).addValidator(StandardValidators.NON_BLANK_VALIDATOR)
				.build();
	}

	private UserItemPair createPairFromContext(ProcessContext context, FlowFile flowfile) {

		Integer userId = Integer
				.parseInt(context.getProperty(USER_ID).evaluateAttributeExpressions(flowfile).getValue());
		Integer itemId = Integer
				.parseInt(context.getProperty(ITEM_ID).evaluateAttributeExpressions(flowfile).getValue());
		UserItemPair pair = new UserItemPair(userId, itemId);
		return pair;

	}

	@OnScheduled
	public void initModel(final ProcessContext context) {
		rcm = new NueralCFModel();
		String modelPath = context.getProperty(MODEL_FILE_PATH).getValue(); 
		rcm.load(modelPath);
	}

	@Override
	public void onTrigger(ProcessContext context, final ProcessSession session) throws ProcessException {

		FlowFile flowFile = session.get();
		if (flowFile == null) {
			return;
		}

		final ComponentLog log = getLogger();
		UserItemPair pair = createPairFromContext(context, flowFile);
		List<UserItemPair> userItemPairs = new ArrayList<UserItemPair>();
		userItemPairs.add(pair);
		// getLogger().info(payload);
		try {
			List<List<JTensor>> jts = rcm.preProcess(userItemPairs);
			List<List<JTensor>> finalResult = rcm.predict(jts);
			log.debug("Predict Result : " + finalResult.get(0).get(0).getData()[0]);
			if (finalResult.isEmpty()) {
				session.transfer(flowFile, REL_FAILURE);
			} else {
				try {
					flowFile = session.putAttribute(flowFile, "score",
							Float.toString(finalResult.get(0).get(0).getData()[0]));
					session.transfer(flowFile, REL_SUCCESS);
				} catch (Exception e) {
					log.error("Returned an error" + e.getMessage());
					flowFile = session.penalize(flowFile);
					session.transfer(flowFile, REL_FAILURE);
				}
			}
		} catch (Exception e) {
			log.error("Failure processing flowfile {} due to {}, penalizing and routing to failure",
					new Object[] { flowFile, e.getMessage() }, e);
			flowFile = session.penalize(flowFile);
			session.transfer(flowFile, REL_FAILURE);
		}
	}

}