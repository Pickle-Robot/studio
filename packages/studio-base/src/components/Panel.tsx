// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/
//
// This file incorporates work covered by the following copyright and
// permission notice:
//
//   Copyright 2018-2021 Cruise LLC
//
//   This source code is licensed under the Apache License, Version 2.0,
//   found at http://www.apache.org/licenses/LICENSE-2.0
//   You may not use this file except in compliance with the License.

import { MessageBar, MessageBarType, Stack, useTheme } from "@fluentui/react";
import BorderAllIcon from "@mdi/svg/svg/border-all.svg";
import CloseIcon from "@mdi/svg/svg/close.svg";
import ExpandAllOutlineIcon from "@mdi/svg/svg/expand-all-outline.svg";
import FullscreenIcon from "@mdi/svg/svg/fullscreen.svg";
import GridLargeIcon from "@mdi/svg/svg/grid-large.svg";
import TrashCanOutlineIcon from "@mdi/svg/svg/trash-can-outline.svg";
import cx from "classnames";
import { last } from "lodash";
import React, {
  useState,
  useCallback,
  useContext,
  useMemo,
  useRef,
  ComponentType,
  Profiler,
  MouseEventHandler,
  useLayoutEffect,
} from "react";
import DocumentEvents from "react-document-events";
import {
  MosaicContext,
  MosaicWindowActions,
  MosaicWindowContext,
  getNodeAtPath,
  getOtherBranch,
  updateTree,
  MosaicNode,
} from "react-mosaic-component";
import { useMountedState } from "react-use";
import styled from "styled-components";

import { useShallowMemo } from "@foxglove/hooks";
import { useConfigById } from "@foxglove/studio-base/PanelAPI";
import Button from "@foxglove/studio-base/components/Button";
import ErrorBoundary, { ErrorRendererProps } from "@foxglove/studio-base/components/ErrorBoundary";
import Flex from "@foxglove/studio-base/components/Flex";
import Icon from "@foxglove/studio-base/components/Icon";
import KeyListener from "@foxglove/studio-base/components/KeyListener";
import { LegacyButton } from "@foxglove/studio-base/components/LegacyStyledComponents";
import PanelContext from "@foxglove/studio-base/components/PanelContext";
import PanelToolbar from "@foxglove/studio-base/components/PanelToolbar";
import {
  useCurrentLayoutActions,
  useSelectedPanels,
} from "@foxglove/studio-base/context/CurrentLayoutContext";
import { usePanelCatalog } from "@foxglove/studio-base/context/PanelCatalogContext";
import { useWorkspace } from "@foxglove/studio-base/context/WorkspaceContext";
import usePanelDrag from "@foxglove/studio-base/hooks/usePanelDrag";
import { TabPanelConfig } from "@foxglove/studio-base/types/layouts";
import { PanelConfig, SaveConfig, PanelConfigSchema } from "@foxglove/studio-base/types/panels";
import { TAB_PANEL_TYPE } from "@foxglove/studio-base/util/globalConstants";
import {
  getPanelIdForType,
  getPanelTypeFromId,
  getPathFromNode,
  updateTabPanelLayout,
} from "@foxglove/studio-base/util/layout";
import { colors } from "@foxglove/studio-base/util/sharedStyleConstants";

import styles from "./Panel.module.scss";

const PerfInfo = styled.div`
  position: absolute;
  white-space: pre-line;
  bottom: 2px;
  left: 2px;
  font-size: 9px;
  opacity: 0.7;
  user-select: none;
  mix-blend-mode: difference;
`;

type Props<Config> = {
  childId?: string;
  overrideConfig?: Config;
  tabId?: string;
};

export interface PanelStatics<Config> {
  panelType: string;
  defaultConfig: Config;
  supportsStrictMode?: boolean;
  configSchema?: PanelConfigSchema<Config>;
}

// Like React.ComponentType<P>, but without restrictions on the constructor return type.
type ComponentConstructorType<P> = { displayName?: string } & (
  | { new (props: P): React.Component<unknown, unknown> }
  | { (props: P): React.ReactElement<unknown> | ReactNull }
);

function ErrorToolbar(errorProps: ErrorRendererProps): JSX.Element {
  const theme = useTheme();
  return (
    <Stack style={{ overflow: "hidden" }}>
      <PanelToolbar backgroundColor={theme.semanticColors.errorBackground}>
        <MessageBar
          messageBarType={MessageBarType.error}
          dismissIconProps={{ iconName: "Refresh" }}
          dismissButtonAriaLabel="Reset"
          onDismiss={errorProps.onDismiss}
        >
          {errorProps.error.toString()}
        </MessageBar>
      </PanelToolbar>
      {errorProps.defaultRenderErrorDetails(errorProps)}
    </Stack>
  );
}

// HOC that wraps panel in an error boundary and flex box.
// Gives panel a `config` and `saveConfig`.
//   export default Panel(MyPanelComponent)
//
// `config` comes from the current layout, but in stories / tests you can pass in your own:
//   `<MyPanel config={…} />`
export default function Panel<
  Config extends PanelConfig,
  PanelProps extends { config: Config; saveConfig: SaveConfig<Config> },
>(
  PanelComponent: ComponentConstructorType<PanelProps> & PanelStatics<Config>,
): ComponentType<Props<Config> & Omit<PanelProps, "config" | "saveConfig">> & PanelStatics<Config> {
  function ConnectedPanel(props: Props<Config>) {
    const { childId, overrideConfig, tabId, ...otherProps } = props;

    const isMounted = useMountedState();

    const { mosaicActions } = useContext(MosaicContext);
    const { mosaicWindowActions }: { mosaicWindowActions: MosaicWindowActions } =
      useContext(MosaicWindowContext);

    const {
      selectedPanelIds,
      setSelectedPanelIds,
      selectAllPanels,
      togglePanelSelected,
      getSelectedPanelIds,
    } = useSelectedPanels();

    const isSelected = useMemo(
      () => childId != undefined && selectedPanelIds.includes(childId),
      [childId, selectedPanelIds],
    );
    const numSelectedPanelsIfSelected = useMemo(
      () => (isSelected ? selectedPanelIds.length : 0),
      [isSelected, selectedPanelIds],
    );

    const {
      savePanelConfigs,
      updatePanelConfigs,
      createTabPanel,
      closePanel,
      getCurrentLayoutState,
    } = useCurrentLayoutActions();

    const [quickActionsKeyPressed, setQuickActionsKeyPressed] = useState(false);
    const [shiftKeyPressed, setShiftKeyPressed] = useState(false);
    const [cmdKeyPressed, setCmdKeyPressed] = useState(false);
    const [fullScreen, setFullScreen] = useState(false);
    const [fullScreenLocked, setFullScreenLocked] = useState(false);
    const panelCatalog = usePanelCatalog();

    const type = PanelComponent.panelType;
    const title = useMemo(
      () => panelCatalog.getPanelByType(type)?.title ?? "",
      [panelCatalog, type],
    );

    const defaultConfig = PanelComponent.defaultConfig;
    const [savedConfig, saveConfig] = useConfigById<Config>(childId);

    // PanelSettings needs useConfigById to return a config
    // If there is no saved config, we save the default config provided by the panel.
    // This typically happens when a new panel is added and the layout does not yet have a config.
    // Even if this effect gets run more than once, we only need to save the default config once.
    const savedDefaultConfig = useRef(false);
    useLayoutEffect(() => {
      if (!savedConfig && !savedDefaultConfig.current) {
        savedDefaultConfig.current = true;
        saveConfig(defaultConfig);
      }
    }, [defaultConfig, saveConfig, savedConfig]);

    const panelComponentConfig = useMemo(
      () => ({ ...defaultConfig, ...savedConfig, ...overrideConfig }),
      [savedConfig, defaultConfig, overrideConfig],
    );

    // Open a panel next to the current panel, of the specified `panelType`.
    // If such a panel already exists, we update it with the new props.
    const openSiblingPanel = useCallback(
      async (panelType: string, siblingConfigCreator: (config: PanelConfig) => PanelConfig) => {
        const panelsState = getCurrentLayoutState().selectedLayout?.data;
        if (!panelsState) {
          return;
        }
        const siblingPanel = panelCatalog.getPanelByType(panelType);
        if (!siblingPanel) {
          return;
        }

        const siblingModule = await siblingPanel.module();
        if (!isMounted()) {
          return;
        }

        const siblingDefaultConfig = siblingModule.default.defaultConfig;
        const ownPath = mosaicWindowActions.getPath();

        // Try to find a sibling panel and update it with the `siblingConfig`
        const lastNode = last(ownPath);
        const siblingPathEnd = lastNode != undefined ? getOtherBranch(lastNode) : "second";
        const siblingPath = ownPath.slice(0, -1).concat(siblingPathEnd);
        const siblingId = getNodeAtPath(mosaicActions.getRoot(), siblingPath);
        if (typeof siblingId === "string" && getPanelTypeFromId(siblingId) === panelType) {
          const siblingConfig: PanelConfig = {
            ...siblingDefaultConfig,
            ...panelsState.configById[siblingId],
          };
          savePanelConfigs({
            configs: [
              {
                id: siblingId,
                config: siblingConfigCreator(siblingConfig),
                defaultConfig: siblingDefaultConfig,
              },
            ],
          });
          return;
        }

        // Otherwise, open new panel
        const newPanelPath = ownPath.concat("second");
        void mosaicWindowActions.split({ type: panelType }).then(() => {
          const newPanelId = getNodeAtPath(mosaicActions.getRoot(), newPanelPath) as string;
          savePanelConfigs({
            configs: [
              {
                id: newPanelId,
                config: siblingConfigCreator(siblingDefaultConfig),
                defaultConfig: siblingDefaultConfig,
              },
            ],
          });
        });
      },
      [
        getCurrentLayoutState,
        isMounted,
        mosaicActions,
        mosaicWindowActions,
        panelCatalog,
        savePanelConfigs,
      ],
    );

    const { panelSettingsOpen } = useWorkspace();

    const onOverlayClick: MouseEventHandler<HTMLDivElement> = useCallback(
      (e) => {
        if (!fullScreen && quickActionsKeyPressed) {
          setFullScreen(true);
          if (shiftKeyPressed) {
            setFullScreenLocked(true);
          }
          return;
        }

        if (childId == undefined) {
          return;
        }
        if (panelSettingsOpen) {
          // Allow clicking with no modifiers to select a panel (and deselect others) when panel settings are open
          e.stopPropagation(); // select the deepest clicked panel, not parent tab panels
          setSelectedPanelIds(isSelected ? [] : [childId]);
        } else if (e.metaKey || shiftKeyPressed || isSelected) {
          e.stopPropagation(); // select the deepest clicked panel, not parent tab panels
          togglePanelSelected(childId, tabId);
        }
      },
      [
        childId,
        tabId,
        fullScreen,
        quickActionsKeyPressed,
        togglePanelSelected,
        shiftKeyPressed,
        isSelected,
        setSelectedPanelIds,
        panelSettingsOpen,
      ],
    );

    const groupPanels = useCallback(() => {
      const layout = getCurrentLayoutState().selectedLayout?.data?.layout;
      if (layout == undefined) {
        return;
      }
      createTabPanel({
        idToReplace: childId,
        layout,
        idsToRemove: getSelectedPanelIds(),
        singleTab: true,
      });
    }, [getCurrentLayoutState, getSelectedPanelIds, createTabPanel, childId]);

    const createTabs = useCallback(() => {
      const layout = getCurrentLayoutState().selectedLayout?.data?.layout;
      if (layout == undefined) {
        return;
      }
      createTabPanel({
        idToReplace: childId,
        layout,
        idsToRemove: getSelectedPanelIds(),
        singleTab: false,
      });
    }, [getCurrentLayoutState, getSelectedPanelIds, createTabPanel, childId]);

    const removePanel = useCallback(() => {
      closePanel({
        path: mosaicWindowActions.getPath(),
        root: mosaicActions.getRoot() as MosaicNode<string>,
        tabId,
      });
    }, [closePanel, mosaicActions, mosaicWindowActions, tabId]);

    const splitPanel = useCallback(() => {
      const savedProps = getCurrentLayoutState().selectedLayout?.data?.configById;
      if (!savedProps) {
        return;
      }
      const tabSavedProps = tabId != undefined ? (savedProps[tabId] as TabPanelConfig) : undefined;
      if (tabId != undefined && tabSavedProps != undefined && childId != undefined) {
        const newId = getPanelIdForType(PanelComponent.panelType);
        const activeTabLayout = tabSavedProps.tabs[tabSavedProps.activeTabIdx]?.layout;
        if (activeTabLayout == undefined) {
          return;
        }
        const pathToPanelInTab = getPathFromNode(childId, activeTabLayout);
        const newTabLayout = updateTree(activeTabLayout, [
          {
            path: pathToPanelInTab,
            spec: { $set: { first: childId, second: newId, direction: "row" } },
          },
        ]);
        const newTabConfig = updateTabPanelLayout(newTabLayout, tabSavedProps);
        savePanelConfigs({
          configs: [
            { id: tabId, config: newTabConfig },
            { id: newId, config: panelComponentConfig },
          ],
        });
      } else {
        void mosaicWindowActions.split({ type: PanelComponent.panelType });
      }
    }, [
      childId,
      getCurrentLayoutState,
      mosaicWindowActions,
      panelComponentConfig,
      savePanelConfigs,
      tabId,
    ]);

    const { onMouseMove, enterFullscreen, exitFullScreen } = useMemo(
      () => ({
        onMouseMove: ((e) => {
          if (e.metaKey !== cmdKeyPressed) {
            setCmdKeyPressed(e.metaKey);
          }
        }) as MouseEventHandler<HTMLDivElement>,
        enterFullscreen: () => {
          setFullScreen(true);
          setFullScreenLocked(true);
        },
        exitFullScreen: () => {
          setFullScreen(false);
          setFullScreenLocked(false);
        },
      }),
      [cmdKeyPressed],
    );

    const onReleaseQuickActionsKey = useCallback(() => {
      setQuickActionsKeyPressed(false);
      if (fullScreen && !fullScreenLocked) {
        exitFullScreen();
      }
    }, [exitFullScreen, fullScreen, fullScreenLocked]);

    const { keyUpHandlers, keyDownHandlers } = useMemo(
      () => ({
        keyUpHandlers: {
          "`": () => onReleaseQuickActionsKey(),
          "~": () => onReleaseQuickActionsKey(),
          Shift: () => setShiftKeyPressed(false),
          Meta: () => setCmdKeyPressed(false),
        },
        keyDownHandlers: {
          a: (e: KeyboardEvent) => {
            e.preventDefault();
            if (cmdKeyPressed) {
              selectAllPanels();
            }
          },
          "`": () => setQuickActionsKeyPressed(true),
          "~": () => setQuickActionsKeyPressed(true),
          Shift: () => setShiftKeyPressed(true),
          Escape: () => exitFullScreen(),
          Meta: () => setCmdKeyPressed(true),
        },
      }),
      [selectAllPanels, cmdKeyPressed, exitFullScreen, onReleaseQuickActionsKey],
    );

    const onBlurDocument = useCallback(() => {
      exitFullScreen();
      setCmdKeyPressed(false);
      setShiftKeyPressed(false);
      onReleaseQuickActionsKey();
    }, [exitFullScreen, onReleaseQuickActionsKey]);

    const otherPanelProps = useShallowMemo(otherProps);
    const childProps = useMemo(
      // We have to lie to TypeScript with "as PanelProps" because the "PanelProps extends {...}"
      // constraint technically allows the panel to require the types of config/saveConfig be more
      // specific types that aren't satisfied by the functions we pass in
      () => ({ config: panelComponentConfig, saveConfig, ...otherPanelProps } as PanelProps),
      [otherPanelProps, panelComponentConfig, saveConfig],
    );
    const child = useMemo(() => <PanelComponent {...childProps} />, [childProps]);

    const renderCount = useRef(0);

    const perfInfo = useRef<HTMLDivElement>(ReactNull);
    const quickActionsOverlayRef = useRef<HTMLDivElement>(ReactNull);
    const onDragStart = useCallback(() => {
      // Temporarily hide the overlay so that the panel can be shown as the drag preview image --
      // even though the overlay is a sibling rather than a child, Chrome still includes it in the
      // preview if it is visible. Changing the appearance in the next React render cycle is not
      // enough; it actually needs to happen during the dragstart event.
      // https://bugs.chromium.org/p/chromium/issues/detail?id=1203107
      const overlay = quickActionsOverlayRef.current;
      if (overlay) {
        overlay.style.opacity = "0";
        setTimeout(() => (overlay.style.opacity = "1"));
      }
    }, []);
    const dragSpec = { tabId, panelId: childId, onDragStart };
    const [connectOverlayDragSource, connectOverlayDragPreview] = usePanelDrag(dragSpec);
    const [connectToolbarDragHandle, connectToolbarDragPreview] = usePanelDrag(dragSpec);

    return (
      <Profiler
        id={childId ?? "$unknown_id"}
        onRender={(
          _id,
          _phase,
          actualDuration,
          _baseDuration,
          _startTime,
          _commitTime,
          _interactions,
        ) => {
          if (perfInfo.current) {
            perfInfo.current.innerText = `${++renderCount.current}\n${actualDuration.toFixed(1)}ms`;
          }
        }}
      >
        <PanelContext.Provider
          value={{
            type,
            id: childId ?? "$unknown_id",
            title,
            config: panelComponentConfig,
            saveConfig: saveConfig as SaveConfig<PanelConfig>,
            updatePanelConfigs,
            openSiblingPanel,
            enterFullscreen,
            hasSettings: PanelComponent.configSchema != undefined,
            tabId,
            supportsStrictMode: PanelComponent.supportsStrictMode ?? true,
            connectToolbarDragHandle,
          }}
        >
          {/* Ensure user exits full-screen mode when leaving window, even if key is still pressed down */}
          <DocumentEvents target={window} enabled onBlur={onBlurDocument} />
          <KeyListener global keyUpHandlers={keyUpHandlers} keyDownHandlers={keyDownHandlers} />
          <Flex
            onClick={onOverlayClick}
            onMouseMove={onMouseMove}
            className={cx({
              [styles.root!]: true,
              [styles.rootFullScreen!]: fullScreen,
              [styles.selected!]: isSelected,
            })}
            col
            dataTest={`panel-mouseenter-container ${childId ?? ""}`}
            clip
            ref={(el) => {
              connectOverlayDragPreview(el);
              connectToolbarDragPreview(el);
            }}
          >
            {isSelected && !fullScreen && numSelectedPanelsIfSelected > 1 && (
              <div data-tab-options className={styles.tabActionsOverlay}>
                <Button style={{ backgroundColor: colors.BLUE }} onClick={groupPanels}>
                  <Icon size="small" style={{ marginBottom: 5 }}>
                    <BorderAllIcon />
                  </Icon>
                  Group in tab
                </Button>
                <Button style={{ backgroundColor: colors.BLUE }} onClick={createTabs}>
                  <Icon size="small" style={{ marginBottom: 5 }}>
                    <ExpandAllOutlineIcon />
                  </Icon>
                  Create {numSelectedPanelsIfSelected} tabs
                </Button>
              </div>
            )}
            {type !== TAB_PANEL_TYPE && quickActionsKeyPressed && !fullScreen && (
              <div
                className={styles.quickActionsOverlay}
                ref={(el) => {
                  quickActionsOverlayRef.current = el;
                  connectOverlayDragSource(el);
                }}
                data-panel-overlay
              >
                <div>
                  <div>
                    <FullscreenIcon />
                    {shiftKeyPressed ? "Lock fullscreen" : "Fullscreen (Shift+click to lock)"}
                  </div>
                  <div>
                    <Button onClick={removePanel}>
                      <TrashCanOutlineIcon />
                      Remove
                    </Button>
                    <Button onClick={splitPanel}>
                      <GridLargeIcon />
                      Split
                    </Button>
                  </div>
                </div>
              </div>
            )}
            {fullScreen && (
              <LegacyButton
                className={styles.exitFullScreen}
                onClick={exitFullScreen}
                data-panel-overlay-exit
              >
                <CloseIcon /> <span>Exit fullscreen</span>
              </LegacyButton>
            )}
            <ErrorBoundary renderError={(errorProps) => <ErrorToolbar {...errorProps} />}>
              {PanelComponent.supportsStrictMode ?? true ? (
                <React.StrictMode>{child}</React.StrictMode>
              ) : (
                child
              )}
            </ErrorBoundary>
            {process.env.NODE_ENV !== "production" && <PerfInfo ref={perfInfo} />}
          </Flex>
        </PanelContext.Provider>
      </Profiler>
    );
  }

  return Object.assign(React.memo(ConnectedPanel), {
    defaultConfig: PanelComponent.defaultConfig,
    panelType: PanelComponent.panelType,
    displayName: `Panel(${PanelComponent.displayName ?? PanelComponent.name})`,
    configSchema: PanelComponent.configSchema,
  });
}
